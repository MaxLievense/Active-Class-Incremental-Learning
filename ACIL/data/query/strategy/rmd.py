import numpy as np
from pytorch_ood.detector.rmd import RMD as rmd

from ACIL.data.query.strategy.base import BaseStrategy


class RMD(rmd, BaseStrategy):
    use_fit = True
    use_features = True
    use_model = True

    def __init__(self, model, name, score_adjustment, **cfg) -> None:
        model = model.network.forward_backbone
        self.score_adjustment = score_adjustment
        super().__init__(model=model, **cfg)

    def fit_features(self, features, labels, device):
        return super().fit_features(z=features, y=labels, device=device)

    def predict_features(self, features):
        return super().predict_features(z=features) * self.score_adjustment

    def select(self, scores: np, n_samples: int) -> np:
        """
        Higher dist is better.
        """
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
