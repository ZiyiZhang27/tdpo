import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import LoRALinearLayer


class NeuronRecycler:
    """
    Recycle the weights connected to dormant neurons, considering the feature map between two network layers as the
    state of a neuron.
    """

    def __init__(self, accelerator, dormant_threshold=0.0, input_size=768):
        super().__init__()
        self.accelerator = accelerator
        self.dormant_threshold = dormant_threshold
        self.activations = {}
        self.handles = []
        self.prev_masks = {}
        self.input_size = input_size

    def _get_activations(self, name):
        """Fetch and store the activations of a network layer."""

        def hook(layer, input, output):
            """
            Get the activations of a layer with relu nonlinearity.
            ReLU has to be called explicitly here because the hook is attached to the Linear layer.
            """
            self.activations[name] = F.relu(output).detach()
            # self.activations[name] = F.leaky_relu(output).detach()

        return hook

    def _register_activation_hook(self, model, use_lora):
        """Register hooks for all Linear layers to calculate activations."""

        self.activations = {}
        if use_lora:
            # only register for the first Linear layer within each LoRALinearLayer
            for name, module in model.named_modules():
                if isinstance(module, LoRALinearLayer):
                    self.handles.append(module.down.register_forward_hook(self._get_activations(name)))
        else:
            for name, module in list(model.named_modules())[:-1]:
                if isinstance(module, nn.Linear):
                    self.handles.append(module.register_forward_hook(self._get_activations(name)))

    def _remove_activation_hook(self):
        """Remove activation hooks for all Linear layers."""

        for handle in self.handles:
            handle.remove()

        self.handles = []

    def _check_hooks_removed(self, model, use_lora):
        """Check whether the activation hook for each Linear layer is removed or not."""

        if use_lora:
            for name, module in model.named_modules():
                if isinstance(module, LoRALinearLayer):
                    if module.down._forward_hooks:
                        print(f"Hooks not removed for {name}")
                    else:
                        print(f"Hooks successfully removed for {name}")
        else:
            for name, module in list(model.named_modules())[:-1]:
                if isinstance(module, nn.Linear):
                    if module._forward_hooks:
                        print(f"Hooks not removed for {name}")
                    else:
                        print(f"Hooks successfully removed for {name}")

    def _get_neuron_masks(self, use_lora, extra_dormant_threshold):
        """
        Compute neuron masks for a given set of activations.
        The returned mask has True where neurons are dormant and False where they are active.
        """

        masks = {}
        extra_masks = {}
        for name, activation in self.activations.items():
            # Gather activations across all processes and concatenate them on the first dimension
            activation = self.accelerator.gather(activation)

            # Taking the mean here conforms to the expectation under D in the main paper's formula
            if use_lora:
                score = activation.abs().mean(dim=(0, 1))
            else:
                score = activation.abs().mean(dim=0)

            # Normalize the score to make the threshold independent of the layer size
            normalized_score = score / (score.mean() + 1e-9)

            # Compute the neuron mask by comparing the normalized score with the threshold
            masks[name] = normalized_score <= self.dormant_threshold

            if extra_dormant_threshold is not None:
                extra_masks[name] = normalized_score <= extra_dormant_threshold

        return masks, extra_masks

    def _get_intersected_percentage(self, masks):
        """Compute the percentage of the intersected dormant neurons with last logging/reset step"""

        if self.prev_masks:
            min_dormant_neurons = 0
            intersected_neurons = 0
            for prev_mask, mask in zip(self.prev_masks.values(), masks.values(), strict=True):
                intersected_mask = prev_mask & mask
                intersected_neurons += torch.numel(intersected_mask)
                min_dormant_neurons += min(torch.numel(prev_mask), torch.numel(mask))
            intersected_percentage = (intersected_neurons / min_dormant_neurons) * 100 if min_dormant_neurons else 0.0
        else:
            intersected_percentage = 0.0

        return intersected_percentage

    def _align_layer_mask(self, model, masks):
        """Align masks with the order of corresponding layers."""

        layers = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinearLayer)}
        masks = {key: masks[key] for key in layers.keys()}

        return masks

    def _get_neuron_stats(self, masks):
        """Compute the statistics of neurons for the given neuron masks."""

        total_neurons = sum([torch.numel(mask) for mask in masks.values()])
        dormant_neurons = sum([torch.sum(mask) for mask in masks.values()])
        dormant_percentage = (dormant_neurons / total_neurons) * 100

        return total_neurons, dormant_neurons, dormant_percentage

    def _linear_layer_reinit(self, layer, mask):
        """Re-initialize the masked neurons of a Linear layer."""

        layer.weight[mask] = nn.init.normal_(layer.weight[mask], mean=0.0, std=1.0 / (self.input_size + 1))
        if layer.bias is not None:
            layer.bias[mask] = nn.init.zeros_(layer.bias[mask])

    def _lora_layer_reinit(self, layer, mask):
        """Re-initialize the masked neurons of a LoRALinearLayer."""

        layer.down.weight[mask] = nn.init.normal_(layer.down.weight[mask], mean=0.0, std=1 / layer.rank)
        layer.up.weight[:, mask] = nn.init.zeros_(layer.up.weight[:, mask])

    def _reset_masked_neurons(self, model, masks, use_lora):
        """Re-initialize the masked neurons of a model."""

        if use_lora:
            layers = [module for module in model.modules() if isinstance(module, LoRALinearLayer)]
            for layer, mask in zip(layers, masks.values(), strict=True):
                if not torch.any(mask):
                    # No masked neurons in this layer
                    continue
                else:
                    self._lora_layer_reinit(layer, mask)
        else:
            layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
            incoming_layers = layers[:-1]
            outgoing_layers = layers[1:]
            for incoming, outgoing, mask in zip(incoming_layers, outgoing_layers, masks.values(), strict=True):
                if not torch.any(mask):
                    # No masked neurons in this layer
                    continue
                else:
                    # 1. Initialize the incoming weights using the initialization distribution
                    self._linear_layer_reinit(incoming, mask)
                    # 2. Initialize the outgoing weights
                    # outgoing.weight[:, mask] = nn.init.zeros_(outgoing.weight[:, mask])
                    outgoing.weight[:, mask] = nn.init.normal_(outgoing.weight[:, mask], std=1 / (self.input_size + 1))

    def _reset_optimizer_moments(self, optimizer, masks, use_lora):
        """Reset the moments of the optimizer for the masked neurons."""

        for i, mask in enumerate(masks.values()):
            if use_lora:
                # Reset the moments for the incoming weights
                optimizer.state_dict()["state"][i * 2]["exp_avg"][mask] = 0.0
                optimizer.state_dict()["state"][i * 2]["exp_avg_sq"][mask] = 0.0
                # Reset the moments for the outgoing weights
                optimizer.state_dict()["state"][i * 2 + 1]["exp_avg"][:, mask] = 0.0
                optimizer.state_dict()["state"][i * 2 + 1]["exp_avg_sq"][:, mask] = 0.0
            else:
                # Reset the moments for the incoming weights and bias
                optimizer.state_dict()["state"][i * 2]["exp_avg"][mask] = 0.0
                optimizer.state_dict()["state"][i * 2]["exp_avg_sq"][mask] = 0.0
                optimizer.state_dict()["state"][i * 2 + 1]["exp_avg"][mask] = 0.0
                optimizer.state_dict()["state"][i * 2 + 1]["exp_avg_sq"][mask] = 0.0
                # Reset the moments for the outgoing weights
                optimizer.state_dict()["state"][(i + 1) * 2]["exp_avg"][:, mask] = 0.0
                optimizer.state_dict()["state"][(i + 1) * 2]["exp_avg_sq"][:, mask] = 0.0

    def init(self, model, use_lora=False):
        """Initialize the recycler for the given model."""

        self._register_activation_hook(model, use_lora)

    def update(self, model, optimizer, reset=False, use_lora=False, extra_dormant_threshold=None):
        """Perform an update step on the recycler for the given model."""

        self._remove_activation_hook()
        # self._check_hooks_removed(model, use_lora)

        masks, extra_masks = self._get_neuron_masks(use_lora, extra_dormant_threshold)
        self.activations = {}

        total_neurons, dormant_neurons, dormant_percentage = self._get_neuron_stats(masks)
        if use_lora:
            intersected_percentage = 0.0
        else:
            intersected_percentage = self._get_intersected_percentage(masks)
            self.prev_masks = masks

        if reset:
            if use_lora:
                masks = self._align_layer_mask(model, masks)
            # Invert dormant neuron masks to active neuron masks
            masks = {k: ~v for k, v in masks.items()}
            # masks = {k: v.fill_(True) for k, v in masks.items()}
            self._reset_masked_neurons(model, masks, use_lora)
            self._reset_optimizer_moments(optimizer, masks, use_lora)

        if extra_dormant_threshold is not None:
            _, extra_dormant_neurons, extra_dormant_percentage = self._get_neuron_stats(extra_masks)
            return total_neurons, dormant_neurons, dormant_percentage, intersected_percentage, extra_dormant_neurons, extra_dormant_percentage

        return total_neurons, dormant_neurons, dormant_percentage, intersected_percentage
