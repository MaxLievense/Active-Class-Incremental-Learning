from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.base import Base

if TYPE_CHECKING:
    from src.data.data import Data


class Model(torch.nn.Module, Base):
    def __init__(
        self,
        device: torch.device,
        data: Data,
        **cfg: dict,
    ) -> None:
        super().__init__()
        Base.__init__(self, cfg)

        self.log.debug(f"Building model...")
        self.device = device
        self.data = data
        self.n_classes = data.n_classes
        self._init_modules()

    def _init_modules(self):
        self.network = instantiate(DictConfig(self.cfg.network), n_classes=self.n_classes, device=self.device)
        self.network.to(self.device)
        self.log.info(f"Loaded model:\n{self.network}")

        self.criterion = instantiate(DictConfig(self.cfg.criterion), model=self.network)
        self.criterion.to(self.device)
        self.log.info(f"Loaded criterion:\n{self.criterion}")

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.criterion.post_forward(self.network(x))

    def step(self, trainer, data, target):
        if self.network.training:
            with torch.set_grad_enabled(True):
                output = self.network(data)
                loss = self.criterion(output, target)
                loss.backward()
                output = self.criterion.post_forward(output)
                return loss.item(), output
        return None, self.network(data)

    def pretrain(self, trainer):
        pass

    def posttrain(self, trainer):
        pass

    def preepoch(self, trainer):
        torch.cuda.empty_cache()

    def postepoch(self, trainer):
        pass

    def evaluate(self, trainer, data, target):
        raise NotImplementedError

    def preview(self, tariner, data, output, target, loss):
        pass

    def save(self, tag: str = ""):
        raise NotImplementedError

    def load(self, file: str):
        raise NotImplementedError
