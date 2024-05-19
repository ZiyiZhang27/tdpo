from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from tdpo_pytorch.accelerator import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import tdpo_pytorch.prompts
import tdpo_pytorch.rewards
from tdpo_pytorch.running_moments import PerPromptRunningMoments, RunningMoments
from tdpo_pytorch.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
from tdpo_pytorch.critic_model import CriticModel
from tdpo_pytorch.neuron_recycler import NeuronRecycler
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
import copy

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/config_tdpo.py:aesthetic", "Training configuration.")

logger = get_logger(__name__)


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state_dict):
        self.value = state_dict["value"]

    def update(self, current: float, n_steps: int):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        pass


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.project_name, config=config.to_dict(), init_kwargs={"wandb": {"name": config.run_name}}
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet)
    # to half-precision as these weights are only used for inference, keeping weights in full precision is not
    # required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    critic_model = CriticModel(
        dtype=torch.float32,
        device=accelerator.device,
        reward_type=config.reward_name1
    )
    if config.sample.kl_penalty_coef > 0:
        config.sample.reference = True
    ref_model = copy.deepcopy(pipeline.unet) if config.sample.reference else None

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    if config.sample.reference:
        ref_model.requires_grad_(False)

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    if config.sample.reference:
        ref_model.to(accelerator.device, dtype=inference_dtype)

    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        pipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop(0)  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop(0)  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # prepare prompt and reward fn
    prompt_fn = getattr(tdpo_pytorch.prompts, config.prompt_fn)
    reward_fn1 = getattr(tdpo_pytorch.rewards, config.reward_fn1)(inference_dtype, accelerator.device)
    reward_fn2 = getattr(tdpo_pytorch.rewards, config.reward_fn2)(inference_dtype, accelerator.device)
    reward_fn3 = getattr(tdpo_pytorch.rewards, config.reward_fn3)(inference_dtype, accelerator.device)
    reward_fn4 = getattr(tdpo_pytorch.rewards, config.reward_fn4)(inference_dtype, accelerator.device)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # Initialize running moments for reward scaling
    if config.sample.reward_scaling:
        if config.per_prompt_running_moments:
            running_moments = PerPromptRunningMoments(
                config.per_prompt_running_moments.buffer_size,
                config.per_prompt_running_moments.min_count,
                mode="scaling",
            )
            accelerator.register_for_checkpointing(running_moments)
        else:
            running_moments = RunningMoments()

    # Initialize running moments for reward normalization
    if config.sample.reward_norm:
        if config.per_prompt_running_moments:
            running_moments = PerPromptRunningMoments(
                config.per_prompt_running_moments.buffer_size,
                config.per_prompt_running_moments.min_count,
                mode="norm",
            )
            accelerator.register_for_checkpointing(running_moments)
        else:
            running_moments = RunningMoments()
    elif config.sample.temporal_reward_norm_per_prompt:
        running_moments = PerPromptRunningMoments(
            config.per_prompt_running_moments.buffer_size,
            config.per_prompt_running_moments.min_count,
            mode="norm",
        )
        accelerator.register_for_checkpointing(running_moments)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    unet, critic_model.mlp = accelerator.prepare(unet, critic_model.mlp)

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer1 = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    optimizer2 = optimizer_cls(
        critic_model.mlp.parameters(),
        lr=config.train.learning_rate_critic,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    optimizer1, optimizer2 = accelerator.prepare(optimizer1, optimizer2)

    # executor to perform callbacks asynchronously.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Set up the KL controller
    # This helps prevent large divergences in the controller (policy)
    if config.sample.kl_penalty_coef > 0:
        if config.sample.target is not None:
            kl_ctl = AdaptiveKLController(config.sample.init_kl_coef, config.sample.target, config.sample.horizon)
            accelerator.register_for_checkpointing(kl_ctl)
        else:
            kl_ctl = FixedKLController(config.sample.init_kl_coef)

    if config.reward_name1 == "hpsv2":
        input_size = 1024
    else:
        input_size = 768
    neuron_recycler_critic = NeuronRecycler(accelerator, config.sample.dormant_threshold, input_size)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        # When resuming from a checkpoint saved by this pipeline, we first re-run the sampling epochs before continuing
        # training, which is essential for reproducibility.
        from tdpo_pytorch.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
        logger.info(f"Resampling from epoch 0 to epoch {first_epoch - 1}")
        global_step = 0
        for epoch in range(first_epoch):
            epoch_info = {}

            lr1 = optimizer1.param_groups[0]['lr']
            lr2 = optimizer2.param_groups[0]['lr']
            epoch_info["learning_rate"] = lr1
            epoch_info["learning_rate_critic"] = lr2
            epoch_info["epoch"] = epoch

            #################### SAMPLING ####################
            pipeline.unet.eval()
            samples = []
            prompts = []
            for i in tqdm(
                range(config.sample.num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                # generate prompts
                prompts, prompt_metadata = zip(
                    *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
                )

                # encode prompts
                prompt_ids = pipeline.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=pipeline.tokenizer.model_max_length,
                ).input_ids.to(accelerator.device)
                prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

                # sample
                with autocast():
                    images, _, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        eta=config.sample.eta,
                        output_type="pt",
                    )

                latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
                log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps)
                timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

                # compute rewards asynchronously
                rewards1 = executor.submit(reward_fn1, images, prompts, prompt_metadata)
                rewards2 = executor.submit(reward_fn2, images, prompts, prompt_metadata)
                rewards3 = executor.submit(reward_fn3, images, prompts, prompt_metadata)
                rewards4 = executor.submit(reward_fn4, images, prompts, prompt_metadata)
                # yield to make sure reward computation starts
                time.sleep(0)

                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                        "log_probs": log_probs,
                        "rewards1": rewards1,
                        "rewards2": rewards2,
                        "rewards3": rewards3,
                        "rewards4": rewards4,
                    }
                )

            # wait for all rewards to be computed
            for sample in tqdm(
                samples,
                desc="Waiting for rewards",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
                rewards1, _ = sample["rewards1"].result()
                rewards2, _ = sample["rewards2"].result()
                rewards3, _ = sample["rewards3"].result()
                rewards4, _ = sample["rewards4"].result()
                sample["rewards1"] = torch.as_tensor(rewards1, device=accelerator.device)
                sample["rewards2"] = torch.as_tensor(rewards2, device=accelerator.device)
                sample["rewards3"] = torch.as_tensor(rewards3, device=accelerator.device)
                sample["rewards4"] = torch.as_tensor(rewards4, device=accelerator.device)

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                for i, image in enumerate(images):
                    pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                # only log rewards from process 0
                accelerator.log(
                    {
                        "images": [
                            wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward1:.2f} | {reward2:.2f} | {reward3:.2f} | {reward4:.2f}")
                            for i, (prompt, reward1, reward2, reward3, reward4) in enumerate(zip(prompts, rewards1, rewards2, rewards3, rewards4))
                        ],
                    },
                    step=global_step,
                )

            # gather rewards across processes
            rewards1 = accelerator.gather(samples["rewards1"]).cpu().numpy()
            rewards2 = accelerator.gather(samples["rewards2"]).cpu().numpy()
            rewards3 = accelerator.gather(samples["rewards3"]).cpu().numpy()
            rewards4 = accelerator.gather(samples["rewards4"]).cpu().numpy()

            del samples["rewards1"], samples["rewards2"], samples["rewards3"], samples["rewards4"]

            # compute the sample-mean rewards
            epoch_info["reward_queries"] = (epoch + 1) * rewards1.size
            rwd_name1 = config.reward_name1
            rwd_name2 = config.reward_name2
            rwd_name3 = config.reward_name3
            rwd_name4 = config.reward_name4
            epoch_info[rwd_name1] = rewards1
            epoch_info[rwd_name2] = rewards2
            epoch_info[rwd_name3] = rewards3
            epoch_info[rwd_name4] = rewards4
            epoch_info[rwd_name1 + "_mean"] = rewards1.mean()
            epoch_info[rwd_name1 + "_std"] = rewards1.std()
            epoch_info[rwd_name2 + "_mean"] = rewards2.mean()
            epoch_info[rwd_name2 + "_std"] = rewards2.std()
            epoch_info[rwd_name3 + "_mean"] = rewards3.mean()
            epoch_info[rwd_name3 + "_std"] = rewards3.std()
            epoch_info[rwd_name4 + "_mean"] = rewards4.mean()
            epoch_info[rwd_name4 + "_std"] = rewards4.std()

            # log debugging values for each epoch
            accelerator.log(epoch_info, step=global_step)
            del epoch_info

            total_batch_size, num_timesteps = samples["timesteps"].shape
            assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
            assert num_timesteps == config.sample.num_steps
            _ = torch.randperm(total_batch_size, device=accelerator.device)
            _ = torch.stack([torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)])

            global_step += 1

            # load checkpoint for the next epoch
            resume_path = config.resume_from.rsplit('_', 1)[0] + '_' + str(epoch)
            logger.info(f"Resuming from {resume_path}")
            accelerator.load_state(resume_path)

    else:
        first_epoch = 0
        global_step = 0

    from tdpo_pytorch.diffusers_patch.pipeline_with_logprob_tdpo import pipeline_with_logprob

    for epoch in range(first_epoch, config.num_epochs):
        epoch_info = {}

        lr1 = optimizer1.param_groups[0]['lr']
        lr2 = optimizer2.param_groups[0]['lr']
        epoch_info["learning_rate"] = lr1
        epoch_info["learning_rate_critic"] = lr2
        epoch_info["epoch"] = epoch

        #################### SAMPLING ####################
        pipeline.unet.eval()
        critic_model.eval()
        if config.sample.reference:
            ref_model.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
            )

            # encode prompts
            prompt_ids = pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

            # sample
            with autocast():
                run_pipeline = partial(
                    pipeline_with_logprob,
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                    ref_model=ref_model,
                    critic_model=critic_model,
                )
                if config.sample.reference:
                    images, _, latents, log_probs, image_embeds, residuals, ref_log_probs = run_pipeline()
                else:
                    images, _, latents, log_probs, image_embeds, residuals = run_pipeline()

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            image_embeds = torch.stack(image_embeds, dim=1)
            log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps)
            if config.sample.reference:
                ref_log_probs = torch.stack(ref_log_probs, dim=1)  # (batch_size, num_steps)
            residuals = torch.stack(residuals, dim=1)  # (batch_size, num_steps)
            timesteps = pipeline.scheduler.timesteps.repeat(config.sample.batch_size, 1)  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards1 = executor.submit(reward_fn1, images, prompts, prompt_metadata)
            rewards2 = executor.submit(reward_fn2, images, prompts, prompt_metadata)
            rewards3 = executor.submit(reward_fn3, images, prompts, prompt_metadata)
            rewards4 = executor.submit(reward_fn4, images, prompts, prompt_metadata)
            # yield to make sure reward computation starts
            time.sleep(0)

            if config.sample.reference:
                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                        "image_embeds": image_embeds,  # each entry is the image embedding before timestep t
                        "log_probs": log_probs,
                        "ref_log_probs": ref_log_probs,
                        "rewards1": rewards1,
                        "rewards2": rewards2,
                        "rewards3": rewards3,
                        "rewards4": rewards4,
                        "residuals": residuals,
                    }
                )
            else:
                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                        "image_embeds": image_embeds,  # each entry is the image embedding before timestep t
                        "log_probs": log_probs,
                        "rewards1": rewards1,
                        "rewards2": rewards2,
                        "rewards3": rewards3,
                        "rewards4": rewards4,
                        "residuals": residuals,
                    }
                )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards1, _ = sample["rewards1"].result()
            rewards2, _ = sample["rewards2"].result()
            rewards3, _ = sample["rewards3"].result()
            rewards4, _ = sample["rewards4"].result()
            sample["rewards1"] = torch.as_tensor(rewards1, device=accelerator.device)
            sample["rewards2"] = torch.as_tensor(rewards2, device=accelerator.device)
            sample["rewards3"] = torch.as_tensor(rewards3, device=accelerator.device)
            sample["rewards4"] = torch.as_tensor(rewards4, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            # only log rewards from process 0
            accelerator.log(
                {
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward1:.2f} | {reward2:.2f} | {reward3:.2f} | {reward4:.2f}")
                        for i, (prompt, reward1, reward2, reward3, reward4) in enumerate(zip(prompts, rewards1, rewards2, rewards3, rewards4))
                    ],
                },
                step=global_step,
            )

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert total_batch_size == config.sample.batch_size * config.sample.num_batches_per_epoch
        assert num_timesteps == config.sample.num_steps

        # gather rewards and residuals across processes
        rewards1 = accelerator.gather(samples["rewards1"]).cpu().numpy()
        rewards2 = accelerator.gather(samples["rewards2"]).cpu().numpy()
        rewards3 = accelerator.gather(samples["rewards3"]).cpu().numpy()
        rewards4 = accelerator.gather(samples["rewards4"]).cpu().numpy()
        residuals = accelerator.gather(samples["residuals"]).cpu().numpy()

        del samples["rewards1"], samples["rewards2"], samples["rewards3"], samples["rewards4"]

        # compute the sample-mean rewards
        epoch_info["reward_queries"] = (epoch + 1) * rewards1.size
        rwd_name1 = config.reward_name1
        rwd_name2 = config.reward_name2
        rwd_name3 = config.reward_name3
        rwd_name4 = config.reward_name4
        epoch_info[rwd_name1] = rewards1
        epoch_info[rwd_name2] = rewards2
        epoch_info[rwd_name3] = rewards3
        epoch_info[rwd_name4] = rewards4
        epoch_info[rwd_name1 + "_mean"] = rewards1.mean()
        epoch_info[rwd_name1 + "_std"] = rewards1.std()
        epoch_info[rwd_name2 + "_mean"] = rewards2.mean()
        epoch_info[rwd_name2 + "_std"] = rewards2.std()
        epoch_info[rwd_name3 + "_mean"] = rewards3.mean()
        epoch_info[rwd_name3 + "_std"] = rewards3.std()
        epoch_info[rwd_name4 + "_mean"] = rewards4.mean()
        epoch_info[rwd_name4 + "_std"] = rewards4.std()

        if config.sample.reward_scaling:
            if config.per_prompt_running_moments:
                # gather the prompts across processes
                prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
                prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                rewards1 = running_moments.update_rewards(prompts, rewards1)
            else:
                running_moments.update(rewards1)
                rewards1 = rewards1 / (running_moments.std + 1e-8)
            epoch_info[rwd_name1 + "_scaled"] = rewards1

        if config.sample.reward_norm:
            if config.per_prompt_running_moments:
                # gather the prompts across processes
                prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
                prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                rewards1 = running_moments.update_rewards(prompts, rewards1)
            else:
                running_moments.update(rewards1)
                rewards1 = (rewards1 - running_moments.mean) / (running_moments.std + 1e-8)
            epoch_info[rwd_name1 + "_normed"] = rewards1

        del samples["prompt_ids"]

        # clip rewards to the range [-clip_range_reward, clip_range_reward]
        if config.sample.clip_range_reward > 0:
            rewards1 = np.clip(
                rewards1,
                -config.sample.clip_range_reward,
                config.sample.clip_range_reward
            )
            epoch_info[rwd_name1 + "_clipped"] = rewards1

        # initialize return values (these are not the final returns)
        returns = np.zeros((samples_per_epoch, num_timesteps))
        returns[:, -1] += rewards1

        if config.sample.reference:
            # estimate the sample-mean KL-divergence between the training model and the reference model
            log_ratio = samples["log_probs"] - samples["ref_log_probs"]
            mean_kl = log_ratio.sum(1).mean()
            mean_kl = accelerator.gather(mean_kl).cpu().numpy().mean()
            epoch_info["mean_kl"] = mean_kl

            mean_kl_per_step = log_ratio.mean()
            mean_kl_per_step = accelerator.gather(mean_kl_per_step).cpu().numpy().mean()
            epoch_info["mean_kl_per_step"] = mean_kl_per_step

            if config.sample.kl_penalty_coef > 0:
                # compute the KL penalty using log_probs and ref_log_probs
                if config.sample.kl_penalty == "kl":
                    kl_penalty = log_ratio
                elif config.sample.kl_penalty == "abs":
                    kl_penalty = torch.abs(log_ratio)
                elif config.sample.kl_penalty == "square":
                    kl_penalty = 0.5 * (log_ratio ** 2)
                elif config.sample.kl_penalty == "clip":
                    kl_penalty = torch.clamp(log_ratio, min=0.0)
                elif config.sample.kl_penalty == "approx":
                    kl_penalty = (torch.exp(log_ratio) - 1) - log_ratio
                elif config.sample.kl_penalty == "js":
                    m = 0.5 * (samples["log_probs"].exp() + samples["ref_log_probs"].exp())
                    kl_penalty = 0.5 * (torch.clamp(samples["log_probs"] - torch.log(m), min=0.0)
                                        + torch.clamp(samples["ref_log_probs"] - torch.log(m), min=0.0))
                else:
                    kl_penalty = log_ratio
                epoch_info["kl_ctl_value"] = kl_ctl.value
                kl_penalty = -kl_penalty * kl_ctl.value
                kl_penalty = accelerator.gather(kl_penalty).cpu().numpy()

                # add the KL penalty term to the initial returns
                returns += config.sample.kl_penalty_coef * kl_penalty

            del samples["ref_log_probs"]

        for t in reversed(range(num_timesteps)):
            next_return = returns[:, t + 1] if t < num_timesteps - 1 else 0.0
            returns[:, t] += config.sample.gamma * next_return
        temporal_rewards = returns - residuals

        if config.sample.temporal_reward_norm_per_prompt and config.per_prompt_running_moments:
            running_moments.update_temporal_reward(prompts, temporal_rewards)
        elif config.sample.temporal_reward_norm:
            temporal_rewards = (temporal_rewards - temporal_rewards.mean()) / (temporal_rewards.std() + 1e-8)

        epoch_info["temporal_reward_mean"] = temporal_rewards.mean()
        epoch_info["temporal_reward_std"] = temporal_rewards.std()
        epoch_info["residual_mean"] = residuals.mean()

        # ungather temporal rewards and returns
        samples["temporal_rewards"] = (
            torch.as_tensor(temporal_rewards)
            .reshape(accelerator.num_processes, total_batch_size, -1)[accelerator.process_index]
            .to(accelerator.device)
        )
        samples["returns"] = (
            torch.as_tensor(returns)
            .reshape(accelerator.num_processes, total_batch_size, -1)[accelerator.process_index]
            .to(accelerator.device)
        )

        # log debugging values for each epoch
        accelerator.log(epoch_info, step=global_step)
        del epoch_info
        torch.cuda.empty_cache()

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            if config.train.shuffle_batch:
                # shuffle samples along batch dimension
                perm = torch.randperm(total_batch_size, device=accelerator.device)
                samples = {k: v[perm] for k, v in samples.items()}

            if config.train.shuffle_timestep:
                # shuffle along timestep dimension independently for each sample
                perms = torch.stack(
                    [torch.randperm(num_timesteps, device=accelerator.device) for _ in range(total_batch_size)]
                )
                sample_keys = [
                    "timesteps",
                    "latents",
                    "next_latents",
                    "image_embeds",
                    "log_probs",
                    "residuals",
                    "temporal_rewards",
                    "returns",
                ]
                for key in sample_keys:
                    samples[key] = samples[key][torch.arange(total_batch_size, device=accelerator.device)[:, None], perms]

            # rebatch for training
            samples_batched = {k: v.reshape(-1, config.train.batch_size, *v.shape[1:]) for k, v in samples.items()}

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())]

            # train
            pipeline.unet.train()
            critic_model.mlp.train()
            info = defaultdict(list)
            for j in tqdm(
                range(num_train_timesteps),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                # policy update
                for i, sample in tqdm(
                    list(enumerate(samples_batched)),
                    desc="Batch",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    if config.train.cfg:
                        # concat negative prompts to sample prompts to avoid two forward passes
                        embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
                    else:
                        embeds = sample["prompt_embeds"]

                    training_layers = unet, critic_model.mlp

                    with accelerator.accumulate(training_layers):
                        with autocast():
                            if config.train.cfg:
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                                    noise_pred_text - noise_pred_uncond
                                )
                            else:
                                noise_pred = unet(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            # compute the log prob of next_latents given latents under the current model
                            _, log_prob = ddim_step_with_logprob(
                                pipeline.scheduler,
                                noise_pred,
                                sample["timesteps"][:, j],
                                sample["latents"][:, j],
                                eta=config.sample.eta,
                                prev_sample=sample["next_latents"][:, j],
                            )
                            # predict the new residual
                            residual = critic_model.get_residual(sample["image_embeds"][:, j])

                        # compute the importance sampling ratio and the approximate policy KL
                        log_ratio = log_prob - sample["log_probs"][:, j]
                        ratio = torch.exp(log_ratio)
                        approx_kl = torch.mean((ratio - 1) - log_ratio)

                        temporal_rewards = sample["temporal_rewards"][:, j]

                        # policy loss
                        pg_loss1 = -temporal_rewards * ratio
                        pg_loss2 = -temporal_rewards * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range
                        )
                        pg_loss = torch.mean(torch.maximum(pg_loss1, pg_loss2))
                        pg_clipfrac = torch.mean(torch.gt(pg_loss2, pg_loss1).float())

                        # critic loss
                        if config.train.clip_range_critic > 0:
                            residuals_clipped = torch.clamp(
                                residual,
                                sample["residuals"][:, j] - config.train.clip_range_critic,
                                sample["residuals"][:, j] + config.train.clip_range_critic,
                            )
                            cf_loss_unclipped = (residual - sample["returns"][:, j]) ** 2
                            cf_loss_clipped = (residuals_clipped - sample["returns"][:, j]) ** 2
                            cf_loss = 0.5 * torch.mean(torch.maximum(cf_loss_unclipped, cf_loss_clipped))
                            cf_clipfrac = torch.mean(torch.gt(cf_loss_clipped, cf_loss_unclipped).float())
                        else:
                            cf_loss = 0.5 * torch.mean((residual - sample["returns"][:, j]) ** 2)

                        loss = pg_loss + config.train.cf_coef * cf_loss

                        # debugging values
                        info["approx_kl"].append(approx_kl)
                        info["pg_clipfrac"].append(pg_clipfrac)
                        info["cf_clipfrac"].append(cf_clipfrac)
                        info["loss"].append(loss)
                        info["pg_loss"].append(pg_loss)
                        info["cf_loss"].append(cf_loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            # this is a hack to use 'accelerator.clip_grad_norm_()' synchronously for two models.
                            # we remove 'accelerator.unscale_gradients()' from 'accelerator.clip_grad_norm_()'
                            # and add it here.
                            accelerator.unscale_gradients()
                            accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                            accelerator.clip_grad_norm_(critic_model.mlp.parameters(), config.train.max_grad_norm)
                        optimizer1.step()
                        optimizer2.step()
                        optimizer1.zero_grad()
                        optimizer2.zero_grad()

            # log training-related stuff
            info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
            info = accelerator.reduce(info, reduction="mean")
            info["inner_epoch"] = inner_epoch

            # collect the neuron stats
            pipeline.unet.eval()
            critic_model.eval()
            i = config.sample.batch_size
            with autocast():
                with torch.no_grad():
                    neuron_recycler_critic.init(critic_model.mlp)

                    _ = critic_model.get_residual(samples["image_embeds"][:i, num_train_timesteps - 1])

                    if (epoch + 1) % config.sample.neuron_reset_freq == 0:
                        random_state1 = torch.get_rng_state()
                        random_state2 = torch.cuda.get_rng_state_all()
                        torch.manual_seed(config.seed)
                        torch.cuda.manual_seed_all(config.seed)
                        neuron_stats_critic = neuron_recycler_critic.update(
                            critic_model.mlp,
                            optimizer2,
                            reset=config.sample.neuron_reset_critic,
                            extra_dormant_threshold=config.sample.extra_dormant_threshold,
                        )
                        torch.set_rng_state(random_state1)
                        torch.cuda.set_rng_state_all(random_state2)
                        logger.info("Masked neurons are successfully reset !!!")
                    else:
                        neuron_stats_critic = neuron_recycler_critic.update(
                            critic_model.mlp,
                            optimizer2,
                            reset=False,
                            extra_dormant_threshold=config.sample.extra_dormant_threshold,
                        )

            info["total_neurons_critic"] = neuron_stats_critic[0]
            info["dormant_neurons_critic"] = neuron_stats_critic[1]
            info["dormant_percentage_critic"] = neuron_stats_critic[2]
            info["intersected_percentage_critic"] = neuron_stats_critic[3]
            if config.sample.extra_dormant_threshold is not None:
                info["extra_dormant_neurons_critic"] = neuron_stats_critic[4]
                info["extra_dormant_percentage_critic"] = neuron_stats_critic[5]

            accelerator.log(info, step=global_step)
            global_step += 1
            info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

        if config.sample.kl_penalty_coef > 0:
            # update the KL controller by samples_per_epoch
            kl_ctl.update(mean_kl, samples_per_epoch)

        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state()


if __name__ == "__main__":
    app.run(main)
