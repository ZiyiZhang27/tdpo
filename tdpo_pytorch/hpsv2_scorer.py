import os
import torch
from transformers import CLIPProcessor
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer


class HPSv2Scorer(torch.nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

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
        self.tokenizer = get_tokenizer('ViT-H-14')
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(self.device) for k, v in inputs.items()}["pixel_values"]
        text = self.tokenizer(prompts).to(self.device)
        outputs = self.model(inputs, text)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T
        scores = torch.diagonal(logits_per_image)

        return scores
