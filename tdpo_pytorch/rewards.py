import torch


def aesthetic_score(dtype, device):
    from .aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts, metadata):
        scores = scorer(images)
        return scores, {}

    return _fn


def hpsv2(dtype, device):
    from .hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts, metadata):
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def ImageReward(dtype, device):
    from .ImageReward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts, metadata):
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def PickScore(dtype, device):
    from .PickScore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)
    scorer.requires_grad_(False)

    def _fn(images, prompts, metadata):
        scores = scorer(images, prompts)
        return scores, {}

    return _fn
