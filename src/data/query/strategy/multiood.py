import inspect
import time

import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from pytorch_ood.api import Detector, ModelNotSetException
from torch import Tensor

from src.base import Base
from src.utils.parse import match_parameters


class MultiOOD(Detector, Base):
    use_fit = None
    use_features = None
    use_latent = None
    use_model = True
    classifier = None
    labelled_feedback = False

    def __init__(self, model, **cfg) -> None:
        Base.__init__(self, cfg)
        self.model = model
        self.strategies = {}
        self.strategies = {
            strategy_cfg.name: [instantiate(strategy_cfg, model=self.model, _recursive_=False), 0]
            for strategy_cfg in self.cfg.strategies.values()
        }
        self.use_fit = any([strategy.use_fit for strategy, _ in self.strategies.values()])
        self.use_features = any([strategy.use_features for strategy, _ in self.strategies.values()])
        self.use_latent = any([strategy.use_latent for strategy, _ in self.strategies.values()])
        self.log.info(f"MultiOOD strategies: {', '.join(self.strategies.keys())}")
        self.strategy_selections = None
        self.selection = None
        self.selection_logits = None

    def reset(self):
        data = {}
        for name, (strategy, t) in self.strategies.items():
            start = time.time()
            if hasattr(strategy, "reset"):
                strategy.reset()
            data[f"OOD/time/{name}"] = time.time() - start + t
            self.strategies[name][1] = 0
        wandb.log(data, commit=False)

    def fit(self, *args, **kwargs):
        raise NotImplementedError("Query should fit (thus extract_features) in inference()")

    def fit_features(self, features, logits, labels, z, device):
        args = {
            "logits": logits,
            "labels": labels,
            "features": features,
            "z": z,
            "device": device,
        }
        for name, (strategy, t) in self.strategies.items():
            start = time.time()
            match_parameters(strategy.fit_features, args)
            self.strategies[name][1] = t + time.time() - start

    def predict(self, x: Tensor) -> Tensor:
        """
        Caclulate uncertainty for inputs
        """
        if self.model is None:
            raise ModelNotSetException
        features = self.model.network.forward_backbone(x)
        z, logits = self.model.network.forward_head_with_latent(features)
        args = {"features": features, "z": z, "logits": logits}
        scores = torch.zeros((len(self.strategies), z.size(0)))
        for i, (name, (strategy, t)) in enumerate(self.strategies.items()):
            start = time.time()
            if strategy.use_model:
                scores[i] = strategy.predict(x)  # if x is not None else 0
            else:
                scores[i] = match_parameters(strategy.predict_features, args)
            self.strategies[name][1] = t + time.time() - start
        scores = scores.t()
        return Tensor(scores)

    def predict_features(self, logits: Tensor) -> Tensor:
        raise NotImplementedError(
            "MultiOOD needs to be able to get latent, so use predict() instead of predict_features()"
        )

    def score(self, logits: Tensor) -> Tensor:
        raise NotImplementedError("MultiOOD does not score, if needed, a Classifier should be used.")

    def select_per_strategy(self, scores: np, n_samples: int) -> Tensor:
        """
        Returns a tensor of shape (n_strategies, n_samples) with the selected indices per strategy.
        E.g.:   [[0, 1, 2], [3, 4, 5], [6, 7, 8]] for 3 strategies and 9 samples.
                Where [0, 1, 2] are the selected indexes by the 0th strategy.
        """
        strategy_selections = torch.empty((len(self.strategies), n_samples))
        for i, (name, (strategy, t)) in enumerate(self.strategies.items()):
            start = time.time()
            strategy_selections[i] = Tensor(
                strategy.select(scores={index: _scores[i] for index, _scores in scores.items()}, n_samples=n_samples)
            )
            self.strategies[name][1] = t + time.time() - start
        self.strategy_selections = strategy_selections
        return strategy_selections

    def select(self, scores: np, n_samples: int) -> np:
        """
        Provided the strategy_selections. Selects n_samples from the collection of strategy_selections.
        Follow the order where the order of selections is maintained per strategy,
            so pick the first, then the second, etc for each strategy.
        """
        strategy_selections = self.select_per_strategy(scores, n_samples).tolist()
        selection = set()
        if len(strategy_selections[0]) == 0:
            self.log.warning("No strategy selections available.")
            return None

        _loop = True
        while len(selection) < n_samples & _loop:
            _loop = False
            for strategy_selection in strategy_selections:
                if len(strategy_selection) > 0:
                    _loop = True
                    if len(selection) < n_samples:
                        selection.add(strategy_selection.pop(0))
        self.selection = selection
        self.selection_logits = Tensor(np.array([scores[index] for index in selection])).clone().detach()

        return list(selection)

    def match_feedback(self, feedback: np) -> list:
        feedback_mapping = {int(index): value for index, value in zip(self.selection, feedback)}
        matched_feedback = []
        for strategy_selection in self.strategy_selections:
            strategy_feedback = []
            for index in strategy_selection:
                feedback_value = feedback_mapping.get(int(index))
                strategy_feedback.append(feedback_value)
            matched_feedback.append(strategy_feedback)

        data = {}
        for i, name in enumerate(self.strategies.keys()):
            data[f"OOD/x/{name}"] = len(matched_feedback[i]) - matched_feedback[i].count(None)
            data[f"OOD/avg/{name}"] = np.mean([value for value in matched_feedback[i] if value is not None])
            data[f"OOD/0/{name}"] = matched_feedback[i].count(0)
            data[f"OOD/5/{name}"] = sum(1 for value in matched_feedback[i] if value is not None and value < 5)
            data[f"OOD/10/{name}"] = sum(1 for value in matched_feedback[i] if value is not None and value < 10)

        wandb.log(data, commit=False)

        return matched_feedback

    def feedback(self, feedback: np) -> list:
        assert self.strategy_selections is not None, "No latest strategy selections available."
        assert self.selection is not None, "No latest selection available."
        assert self.selection_logits is not None, "No latest selection logits available."

        matched_feedback = self.match_feedback(feedback)

        self.strategy_selections = None
        self.selection = None
        self.selection_logits = None

        return matched_feedback


class MultiOODChecker(MultiOOD):
    def __init__(self, model, **cfg) -> None:
        super().__init__(model, **cfg)
        self.log.info(f"MultiOODChecker initialized, selecting with strategy: {self.cfg.name}")

    def select(self, scores: np, n_samples: int) -> np:
        strategy_idx = list(self.strategies.keys()).index(self.cfg.name)
        return self.strategies[self.cfg.name][0].select(
            scores={index: _scores[strategy_idx] for index, _scores in scores.items()}, n_samples=n_samples
        )

    def feedback(self, feedback: np) -> list:
        pass


class MultiOODCheckerAndSampler(MultiOODChecker):
    def __init__(self, model, **cfg) -> None:
        super().__init__(model, **cfg)
        if self.cfg.sample_strategies == "all":
            self.cfg.sample_strategies = list(self.strategies.keys())
        self.log.info(f"MultiOODCheckerAndSampler initialized, selecting with strategy: {self.cfg.sample_strategies}")

    def select(self, scores: np, n_samples: int) -> np:
        selection = np.empty((0, 2), dtype=int)
        # Loop through each strategy in the sample_strategies list
        for i, strategy_name in enumerate(self.cfg.sample_strategies):
            strategy_idx = list(self.strategies.keys()).index(strategy_name)
            strategy_selection = self.strategies[strategy_name][0].select(
                scores={index: _scores[strategy_idx] for index, _scores in scores.items()}, n_samples=n_samples
            )

            # Add it to a np.array, where the 1st element is the rank ([n_samples .. 0]) and the second element is the index
            strategy_selection = np.array(list(enumerate(strategy_selection[::-1], start=1)))
            selection = np.concatenate((selection, strategy_selection), axis=0)

        # sum selection ranks per index
        unique_selections = np.unique(selection[:, 1])
        summed_selection = {}
        for i in unique_selections:
            summed_selection[i] = np.sum(selection[selection[:, 1] == i][:, 0])
        selection = sorted(summed_selection, key=summed_selection.get, reverse=True)[:n_samples]
        assert len(set(selection)) == n_samples
        return list(selection)