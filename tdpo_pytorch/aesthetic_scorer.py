from importlib import resources
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

ASSETS_PATH = resources.files("tdpo_pytorch.assets")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device, dtype=self.dtype)
        self.mlp = MLP().to(self.device, dtype=self.dtype)

        state_dict = torch.load(ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth"), map_location=self.device)
        self.mlp.load_state_dict(state_dict)
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)

        return self.mlp(embed).squeeze(1)
