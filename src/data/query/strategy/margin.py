import numpy as np
import torch
from pytorch_ood.api import Detector, ModelNotSetException
from torch import Tensor

from src.data.query.strategy.base import BaseStrategy


class Margin(BaseStrategy):
    def __init__(self, model, **kwargs) -> None:
        self.model = model

    def predict(self, x: Tensor) -> Tensor:
        """
        Caclulate uncertainty for inputs
        """
        if self.model is None:
            raise ModelNotSetException

        return self.score(self.model(x))

    def predict_features(self, logits: Tensor) -> Tensor:
        return self.score(logits)

    @staticmethod
    def score(logits: Tensor) -> Tensor:
        """
        :param logits: logits of input
        """
        sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
        margins = sorted_logits[:, 0] - sorted_logits[:, 1]
        return margins * -1

    def select(self, scores: np, n_samples: int) -> np:
        """
        Select n samples with smallest margins, the most relatively uncertain samples.
        Return negative values.
        """
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]
