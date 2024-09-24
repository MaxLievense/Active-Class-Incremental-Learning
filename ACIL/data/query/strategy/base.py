import numpy as np
from pytorch_ood.api import Detector


class BaseStrategy(Detector):
    use_fit = False
    use_model = False
    use_features = False
    use_latent = False

    def select(self, scores: np, n_samples: int) -> np:
        raise NotImplementedError

    def feedback(self, feedback: np) -> None:
        """
        Not required.
        """
        return self

    def fit_features(self, *args, **kwargs):
        return self

    def fit(self, *args, **kwargs):
        return self
