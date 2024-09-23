from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from src.base import Base

if TYPE_CHECKING:
    from logging import Logger


@dataclass
class TrainingResults:
    log: Logger

    losses: list = field(default_factory=list)
    accuracies: list = field(default_factory=list)
    ignore: list = field(default_factory=list)

    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.ignore:
                continue
            try:
                getattr(self, k).append(v)
            except AttributeError:
                self.ignore.append(k)
                self.log.warning(f"Could not find attribute {k} in {self.__class__.__name__}.")

    def save(self, output_dir):
        df = pd.DataFrame({k: v for k, v in self.__dict__.items() if k not in ["log", "ignore"]})
        df.to_csv(f"{output_dir}/{self.__class__.__name__}.csv", index=False)


@dataclass
class EvalResults(TrainingResults):
    coverage: list = field(default_factory=list)
    accuracy: list = field(default_factory=list)
    ood_accuracy: list = field(default_factory=list)


class Results(Base):
    def __init__(self, **cfg):
        super().__init__(cfg)
        self.training = TrainingResults(self.log)
        self.eval = EvalResults(self.log)

    def __call__(self, type, **kwargs):
        getattr(self, type)(**kwargs)

    def save(self):
        self.training.save(self.output_dir)
        self.eval.save(self.output_dir)
