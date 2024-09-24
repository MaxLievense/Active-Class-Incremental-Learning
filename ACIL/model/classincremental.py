from copy import deepcopy

from hydra.utils import instantiate
from omegaconf import DictConfig

from ACIL.model.model import Model
from ACIL.utils.subsets import split_holdout


class ClassIncrementalModel(Model):
    def _init_modules(self):
        self.n_classes = int(self.data._n_classes * (1 + self.data.cfg.extra_classes))
        self.log.debug(f"Initializing model with {self.n_classes} classes...")
        self._init_network()
        self._init_criterion()

    def _init_network(self):
        self.network = instantiate(DictConfig(self.cfg.network), n_classes=self.n_classes, device=self.device)
        self.network.to(self.device)
        self.log.info(f"Loaded model:\n{self.network}")

    def _init_criterion(self):
        self.criterion = instantiate(DictConfig(self.cfg.criterion))
        self.criterion.to(self.device)
        self.log.info(f"Loaded criterion:\n{self.criterion}")

    def pretrain(self, trainer):
        trainer.data.get_query(self)

    def postepoch(self, trainer):
        trainer.data.query(trainer)
        self.update_fc(trainer)

    def update_fc(self, trainer):
        if self.cfg.fc.update and trainer.data._n_classes > self.n_classes:
            self.log.info(f"Adding {trainer.data._n_classes - self.n_classes} new classes to fc.")
            self.n_classes = trainer.data._n_classes
            if self.cfg.reload_model_on_new_classes:
                self._init_network()
            else:
                self.network.make_fc(self.n_classes, transfer_weights=self.cfg.fc.transfer_weights)
            if self.cfg.fc.reinit_optimizer or self.cfg.reload_model_on_new_classes:
                trainer.init_optimizer()


class QueryAfterTrain(ClassIncrementalModel):
    def pretrain(self, trainer):
        trainer.data.get_query(self)

    def postepoch(self, trainer):
        pass

    def posttrain(self, trainer):
        if not self.cfg.query:
            return
        trainer.data.query(trainer, force=True)


class SteppedFrozenModel(ClassIncrementalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_step = -1

    def preepoch(self, trainer):
        step = trainer.step
        if self._last_step == step:
            return
        self._last_step = step
        backbone_params, fc_params = self.network.get_params(split="fc")
        if step == self.cfg.steps[0]:
            for param in backbone_params:
                param.requires_grad = False
            for param in fc_params:
                param.requires_grad = True
            self.log.debug("Training FC.")
        elif step == self.cfg.steps[1]:
            for param in backbone_params:
                param.requires_grad = True
            for param in fc_params:
                param.requires_grad = True
            self.log.debug("Training Backbone and FC.")
        else:
            raise ValueError(f"Step {step} not implemented.")

    def postepoch(self, trainer):
        pass

    def posttrain(self, trainer):
        if not self.cfg.query:
            return
        trainer.data.query(trainer, force=True)
        self.update_fc(trainer)


class HoldoutModel(SteppedFrozenModel):
    def preepoch(self, trainer):
        super().preepoch(trainer)
        if not hasattr(self, "_train_data_indices") or len(trainer.data.train_data.indices) > len(
            self._train_data_indices  # pylint: disable=E0203
        ):
            self._train_data_indices = deepcopy(trainer.data.train_data.indices)  # make copy
        trainer.data.train_data.indices = deepcopy(self._train_data_indices)
        trainer.data.train_loader.dataset.indices, trainer.data.val_loader.dataset.indices = split_holdout(
            trainer.data.train_data,
            self.cfg.holdout,
        )

    def postepoch(self, trainer):
        trainer.data.train_data.indices = deepcopy(self._train_data_indices)


class KFoldModel(SteppedFrozenModel):
    class KFold:
        def __init__(self, cfg: DictConfig):
            self.kfold = instantiate(cfg)

        def split(self, dataset):
            for train_idx, val_idx in self.kfold.split(dataset):
                yield train_idx, val_idx

    def preepoch(self, trainer):
        super().preepoch(trainer)
        if not hasattr(self, "_kfold"):
            self._kfold = self.KFold(self.cfg.kfold)
            self._n_fold = 0

        if not hasattr(self, "_train_data_indices") or len(trainer.data.train_data.indices) > len(
            self._train_data_indices  # pylint: disable=E0203
        ):
            self._train_data_indices = deepcopy(trainer.data.train_data.indices)  # make copy
        trainer.data.train_data.indices = deepcopy(self._train_data_indices)
        if self._n_fold == 0:
            self._kfold_generator = self._kfold.split(trainer.data.train_data)

        train_idx, val_idx = next(self._kfold_generator)
        trainer.data.train_loader.dataset.indices = [self._train_data_indices[i] for i in train_idx]
        trainer.data.val_loader.dataset.indices = [self._train_data_indices[i] for i in val_idx]

        self._n_fold += 1

        if self._n_fold == self.cfg.kfold.n_splits:
            self._n_fold = 0

    def postepoch(self, trainer):
        trainer.data.train_data.indices = deepcopy(self._train_data_indices)

    def posttrain(self, trainer):
        """
        If stopped early, the postepoch will not go
        """
        self.postepoch(trainer)
        return super().posttrain(trainer)
