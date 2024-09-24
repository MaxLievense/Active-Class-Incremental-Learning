from __future__ import annotations

import random
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ACIL.utils.base import Base

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Runner(Base):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)
        self._hydra_cfg = HydraConfig.get()
        self.log.info(f"Overrides:\n{OmegaConf.to_yaml(self._hydra_cfg.overrides.task)}")
        self.log.info(f"Config:\n{OmegaConf.to_yaml(self.cfg)}")
        self.setup()

    def setup(self) -> torch.device:
        _overrides_cfg = {
            k: v
            for override in self._hydra_cfg.overrides.task
            for k, v in [override.split("=") if "=" in override else (override, None)]
        }
        _cfg = {**{"overrides": _overrides_cfg}, **OmegaConf.to_container(self.cfg, resolve=True)}
        wandb.init(
            **self.cfg.wandb,
            config=_cfg,
            settings=wandb.Settings(symlink=False),
        )
        self.set_seeds()
        self.device = torch.device(self.cfg.device)
        self.log.info(f"Device: {self.device}")
        torch.cuda.empty_cache()

    def build(self):
        self.data = instantiate(
            self.cfg.data,
            model_transforms=self.cfg.model.network.get("transforms", []),
            device=self.device,
            _recursive_=False,
        )
        self.model = instantiate(self.cfg.model, device=self.device, data=self.data, _recursive_=False)
        self.trainer = instantiate(
            self.cfg.trainer, device=self.device, model=self.model, data=self.data, _recursive_=False
        )

    def run(self):
        self.trainer.run()

    def set_seeds(self):
        if self.cfg.seed is None:
            seed = np.random.randint(0, 2**8 - 1)
            self.log.warning(f"Setting (dataset) seed to: {seed}")
            self.cfg.seed = seed
        else:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed(self.cfg.seed)
            torch.cuda.manual_seed_all(self.cfg.seed)


@hydra.main(version_base="1.2", config_path=".", config_name="main")
def main(cfg: DictConfig):
    runner = Runner(cfg)
    runner.build()
    runner.run()


if __name__ == "__main__":
    main()
