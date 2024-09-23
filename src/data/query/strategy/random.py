import numpy as np
from torch import Tensor

from src.data.query.strategy.base import BaseStrategy


class Random(BaseStrategy):

    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, x: Tensor) -> Tensor:
        return self.score(x)

    def predict_features(self, logits: Tensor) -> Tensor:
        return self.score(logits)

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        return Tensor(np.random.rand(logits.size(0)))

    def select(self, scores: np, n_samples: int) -> np:
        """
        Select n random samples.
        """
        return sorted(scores, key=scores.get)[:n_samples]
