import numpy as np
import torch
from pytorch_ood.detector.vim import ViM as vim

from src.data.query.strategy.base import BaseStrategy


class ViM(vim, BaseStrategy):
    use_fit = True
    use_features = True

    def __init__(self, model, name, **cfg) -> None:
        backbone = model.network.forward_backbone
        weights = (
            model.network.fc.weight if isinstance(model.network.fc, torch.nn.Linear) else model.network.fc[0].weight
        )
        bias = model.network.fc.bias if isinstance(model.network.fc, torch.nn.Linear) else model.network.fc[0].bias
        super().__init__(model=backbone, w=weights, b=bias, **cfg)

    def predict_features(self, features):
        return super().predict_features(x=features)

    def select(self, scores: np, n_samples: int) -> np:
        """
        Higher is better.
        """
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
