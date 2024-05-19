from importlib import resources
import os
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoModel
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms

ASSETS_PATH = resources.files("tdpo_pytorch.assets")


class MLP(nn.Module):
    def __init__(self, reward_type):
        super().__init__()
        if reward_type == "hpsv2" or reward_type == "PickScore":
            self.input_size = 1024
        else:
            self.input_size = 768
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

        # initial MLP params
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        return self.layers(x)


class CriticModel(nn.Module):
    def __init__(self, dtype, device, reward_type):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.reward_type = reward_type

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        if self.reward_type == "hpsv2":
            self.model, _, _ = create_model_and_transforms(
                'ViT-H-14',
                'laion2B-s32B-b79K',
                precision=self.dtype,
                device=self.device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False
            )
            checkpoint_path = f"{os.path.expanduser('~')}/.cache/hpsv2/HPS_v2_compressed.pt"
            # force download of model via score
            hpsv2.score([], "")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])

        elif self.reward_type == "PickScore":
            checkpoint_path = "yuvalkirstain/PickScore_v1"
            # checkpoint_path = f"{os.path.expanduser('~')}/.cache/PickScore_v1"
            self.model = AutoModel.from_pretrained(checkpoint_path)

        else:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        self.mlp = MLP(reward_type)

    @torch.no_grad()
    def get_image_embed(self, images):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}

        if self.reward_type == "hpsv2":
            embed = self.model.encode_image(inputs["pixel_values"], normalize=True)
        else:
            embed = self.model.get_image_features(**inputs)
            embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

        return embed

    def get_residual(self, embed):
        residual = self.mlp(embed)
        return residual.squeeze(1)
