from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig

from data.data import Data
from model.model import Model
from src.base import Base
from src.utils.metrics import class_metrics

if TYPE_CHECKING:
    from src.data.data import Data
    from src.model.model import Model

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")


class Trainer(Base):
    def __init__(
        self,
        device: torch.device,
        callbacks: dict,
        model: Model,
        data: Data,
        **cfg: dict,
    ) -> None:
        super().__init__(cfg)

        self.log.debug(f"Building trainer...")
        self.device = device
        self.model = model
        self.data = data
        self.step = 0
        self.last_epoch = -1
        self.init_optimizer()
        self.callback = instantiate(DictConfig(callbacks))

        self.log.info(f"Loaded trainer.")

    def init_optimizer(self):
        self.optimizer = instantiate(DictConfig(self.cfg.optimizer), params=self.model.network.parameters())
        self.scheduler = (
            instantiate(DictConfig(self.cfg.scheduler), optimizer=self.optimizer) if self.cfg.get("scheduler") else None
        )

    def run(self):
        self.log.info(f"Running trainer...")
        self.train()

        if self.cfg.eval_at_end:
            self.eval()
        if self.cfg.save_last:
            self.save("end")

    def train(self):
        self.model.network.train()
        self.model.pretrain(self)

        self.log.info(f"Training...")
        for epoch in range(self.cfg.epochs):
            self.last_epoch = epoch
            _breaking = False
            self.model.preepoch(self)

            running_loss = 0.0
            running_accuracy = 0.0
            training_entries = 0
            for step, data in enumerate(self.data.train_loader):
                training_entries += len(data[0])
                self.model.network.train()
                data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                self.optimizer.zero_grad()
                step_loss, logits = self.model.step(self, *data)
                self.optimizer.step()
                running_loss += step_loss
                running_accuracy += ((preds := torch.argmax(logits, dim=1)) == data[1]).to("cpu").sum().float().item()
                if self.cfg.verbose:
                    print(f"Epoch {epoch} step {step} loss: {running_loss:.6f} preds: {preds}", end="\r")
            _lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"]
            self.log.info(
                f"Epoch {epoch+1}/{self.cfg.epochs} | "
                + f"avg loss: {running_loss / training_entries:.6f} "
                + f"(lr: {_lr:.5f}) | "
                + f"avg accuracy: {running_accuracy / training_entries:.2%}"
            )
            accuracy = running_accuracy / training_entries
            if self.callback(
                trainer=self,
                loss=running_loss,
                accuracy=accuracy,
                epoch=epoch,
                outputs=logits,
                data=data,
                lr=_lr,
            ):
                _breaking = True
            self.model.postepoch(self)

            if self.scheduler:
                self.scheduler.step(epoch)

            if _breaking:
                break
        self.model.posttrain(self)

    def eval(self, loader="test"):
        if loader == "test":
            self.log.info("Evaluating...")

        _loader = getattr(self.data, loader + "_loader")
        total_labels = []
        total_logits = []
        self.model.network.eval()
        with torch.no_grad():
            for _, data in enumerate(_loader):
                data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                _, logits = self.model.step(self, *data)
                total_logits.append(logits)
                total_labels.append(data[1])
        closed_results, _ = class_metrics(
            y_true=torch.cat(total_labels),
            y_out=torch.cat(total_logits),
            training_epoch=self.last_epoch,
            data=self.data,
            tag=loader,
        )
        return closed_results

    def save(self, tag=None):
        filename = f"{self.output_dir}/{self.last_epoch}{f'.{tag}' if tag else ''}.pth"
        state_dict = self.model.state_dict()
        state_dict["cfg"] = self.model.network.cfg
        torch.save(state_dict, filename)
        self.log.info(f"Model saved as {filename}")


class SelfTrainer(Trainer):
    def train(self):
        self.model.network.train()
        self.model.pretrain(self)

        self.log.info(f"Training...")
        for epoch in range(self.cfg.epochs):
            self.last_epoch = epoch
            self.model.preepoch(self)

            running_loss = 0.0
            training_entries = 0
            for step, data in enumerate(self.data.train_loader):
                training_entries += len(data[0])
                self.model.network.train()
                self.optimizer.zero_grad()
                data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                step_loss, _ = self.model.step(self, *data)
                self.optimizer.step()
                running_loss += step_loss
                if self.cfg.verbose:
                    print(f"Epoch {epoch} step {step} loss: {running_loss:.6f}", end="\r")
            self.log.info(f"Epoch {epoch+1}/{self.cfg.epochs} " + f"avg loss: {running_loss / training_entries:.6f} ")
            if self.callback(
                trainer=self,
                loss=running_loss,
                accuracy=None,
                epoch=epoch,
                outputs=None,
                data=data,
                lr=self.optimizer.param_groups[0]["lr"],
            ):
                break
            self.model.postepoch(self)

            if self.scheduler:
                self.scheduler.step(epoch)
        self.model.posttrain(self)

    def eval(self):
        self.log.info(f"Evaluating...")
        self.model.network.eval()

        total_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(getattr(self.data, self.cfg.test_on)):
                data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                if i == 0:
                    self.model.preview(data[0])
                loss, _ = self.model.step(self, *data)
                total_loss += loss

        wandb.log({"eval/Loss": total_loss}, commit=False)


class SteppedTrainer(Trainer):
    def train(self):
        for iterations in range(self.cfg.iterations):
            self.model.network.train()
            self.model.pretrain(self)

            self.log.info(f"Training...")
            for step in range(self.cfg.steps):
                self.init_optimizer()
                for epoch in range(self.cfg.epochs):
                    self.step = step
                    self.last_epoch = epoch
                    self.model.preepoch(self)

                    running_loss = 0.0
                    running_accuracy = 0.0
                    training_entries = 0
                    for _step, data in enumerate(self.data.train_loader):
                        training_entries += len(data[0])
                        self.model.network.train()
                        data = [_data.to(self.device) if isinstance(_data, torch.Tensor) else _data for _data in data]
                        self.optimizer.zero_grad()
                        step_loss, logits = self.model.step(self, *data)
                        self.optimizer.step()
                        running_loss += step_loss
                        running_accuracy += (
                            ((preds := torch.argmax(logits, dim=1)) == data[1]).to("cpu").sum().float().item()
                        )
                        if self.cfg.verbose:
                            print(f"Epoch {epoch} step {_step} loss: {running_loss:.6f} preds: {preds}", end="\r")
                    _lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"]
                    self.log.info(
                        f"Iter:{iterations+1}/{self.cfg.iterations} - "
                        + f"Step: {step+1}/{self.cfg.steps} - Epoch {epoch+1}/{self.cfg.epochs} | "
                        + f"avg loss: {running_loss / training_entries:.6f} "
                        + f"(lr: {f'{_lr:.5f}' if not isinstance(_lr, list) else ','.join([f'{__lr:.5f}' for __lr in _lr])}) | "
                        + f"avg accuracy: {running_accuracy / training_entries:.2%}"
                    )
                    accuracy = running_accuracy / training_entries
                    if self.callback(
                        trainer=self,
                        loss=running_loss,
                        accuracy=accuracy,
                        epoch=epoch,
                        outputs=logits,
                        data=data,
                        lr=_lr,
                    ):
                        break
                    self.model.postepoch(self)

                    if self.scheduler:
                        self.scheduler.step(epoch)

            self.model.posttrain(self)


class FinetunedTrained(SteppedTrainer):
    """Variation on SteppedTrainer (1 step) where there are 2 optimizers, one for the fc and one for the backbone, both with their own schedulers"""

    class Schedulers:
        def __init__(self, schedulers: list, optimizer: torch.optim.Optimizer):
            self.schedulers = schedulers
            self.optimizer = optimizer

        def step(self, epoch):
            for scheduler in self.schedulers:
                scheduler.step(epoch)
            lrs = [scheduler.get_last_lr()[0] for scheduler in self.schedulers]
            self.optimizer.param_groups[0]["lr"] = lrs[0]
            self.optimizer.param_groups[1]["lr"] = lrs[1]

        def get_last_lr(self):
            return [[self.optimizer.param_groups[i]["lr"] for i in range(len(self.schedulers))], None]

    def __init__(self, device: torch.device, callbacks: dict, model: Model, data: Data, **cfg: dict) -> None:
        super().__init__(device, callbacks, model, data, **cfg)

    def init_optimizer(self):
        self.fc_params = [param for name, param in self.model.network.named_parameters() if "fc" in name]
        self.backbone_params = [param for name, param in self.model.network.named_parameters() if "fc" not in name]

        opt_cfg = self.cfg.optimizer
        if "_target_" in opt_cfg:
            del opt_cfg["_target_"]
        self.optimizer = torch.optim.Adam(
            params=[
                {"name": "fc", "params": self.fc_params},
                {"name": "backbone", "params": self.backbone_params, "lr": 0},
            ],
            **opt_cfg,
        )

        junk_optimizer = torch.optim.Adam(params=torch.nn.Linear(1, 1).parameters(), **opt_cfg)
        fc_scheduler = (
            instantiate(DictConfig(self.cfg.scheduler.fc), optimizer=junk_optimizer)
            if self.cfg.get("scheduler", {}).get("fc")
            else None
        )
        backbone_scheduler = (
            instantiate(DictConfig(self.cfg.scheduler.backbone), optimizer=junk_optimizer)
            if self.cfg.get("scheduler", {}).get("fc")
            else None
        )
        self.scheduler = self.Schedulers([fc_scheduler, backbone_scheduler], self.optimizer)
