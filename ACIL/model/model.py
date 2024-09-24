from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from ACIL.utils.base import Base

if TYPE_CHECKING:
    from ACIL.data.data import Data


class Model(torch.nn.Module, Base):
    def __init__(
        self,
        device: torch.device,
        data: Data,
        **cfg: dict,
    ) -> None:
        super().__init__()
        Base.__init__(self, cfg)

        self.log.debug("Building model...")
        self.device = device
        self.data = data
        self.n_classes = data.n_classes
        self._init_modules()

    def _init_modules(self):
        self.network = instantiate(DictConfig(self.cfg.network), n_classes=self.n_classes, device=self.device)
        self.network.to(self.device)
        self.log.info(f"Loaded model:\n{self.network}")

        self.criterion = instantiate(DictConfig(self.cfg.criterion))
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

    def pretrain(self, *args, **kwargs):
        pass

    def posttrain(self, *args, **kwargs):
        pass

    def preepoch(self, *args, **kwargs):
        torch.cuda.empty_cache()

    def postepoch(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
