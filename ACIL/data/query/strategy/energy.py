import numpy as np
from pytorch_ood.detector.energy import EnergyBased as energy

from ACIL.data.query.strategy.base import BaseStrategy


class Energy(energy, BaseStrategy):

    def __init__(self, model, name, **cfg) -> None:
        super().__init__(model, **cfg)

    def select(self, scores: np, n_samples: int) -> np:
        """
        Higher energy is better.
        """
        return sorted(scores, key=scores.get, reverse=True)[:n_samples]

    def score(self, logits: np, t: float = 1.0) -> np:  # pylint: disable=W0246
        return super().score(logits, t)
