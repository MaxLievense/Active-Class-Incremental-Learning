import numpy as np
from pytorch_ood.detector.klmatching import KLMatching as klmatching

from src.data.query.strategy.base import BaseStrategy


class KLMatching(klmatching, BaseStrategy):
    use_fit = True

    def __init__(self, model, name, **cfg) -> None:
        super().__init__(model=model, **cfg)

    def predict_features(self, logits):
        p = logits.softmax(dim=1)
        return super().predict_features(p=p)

    def select(self, scores: np, n_samples: int) -> np:
        """
        Higher is better.
        """
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
