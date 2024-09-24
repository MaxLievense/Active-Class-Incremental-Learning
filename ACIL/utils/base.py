from __future__ import annotations

import logging
import os

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


class Base:
    def __init__(self, cfg: DictConfig):
        self.name = self.__module__.rsplit(".", maxsplit=1)[-1]
        self.log = logging.getLogger(self.name)
        self.cfg = cfg if isinstance(cfg, DictConfig) else DictConfig(cfg)
        if "output_dir" in cfg:
            self.output_dir_base = HydraConfig.get().runtime.output_dir
            self.output_dir = os.path.join(self.output_dir_base, cfg["output_dir"])
            os.makedirs(self.output_dir, exist_ok=True)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
