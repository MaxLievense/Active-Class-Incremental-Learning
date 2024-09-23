import os
from copy import deepcopy

import numpy as np
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

import hydra
from src.base import Base
from src.data.transforms.transforms import target_class_mapping
from src.utils.metrics import class_coverage
from src.utils.module import import_from_cfg
from src.utils.subsets import (
    Subset,
    make_LT_dataset,
    split_dataset,
    split_dataset_class_balanced,
)


class Data(Base):
    def __init__(
        self,
        device: torch.device,
        dataset_transforms: list,
        model_transforms: list,
        training_transforms: list,
        **cfg: dict,
    ) -> None:
        super().__init__(cfg)
        self.log.debug(f"Building dataset: {self.cfg.dataset.dataset_name}")

        self.dataset_name = self.cfg.dataset.dataset_name
        self.device = device

        self.get_transform(model_transforms, dataset_transforms, training_transforms)
        self.get_dataset(DictConfig(self.cfg.dataset))
        self.get_dataset_info()
        self.get_loaders(DictConfig(self.cfg.dataloaders))

        self.log.info(
            f"Loaded {self.dataset_name} dataset with {self.n_classes} classes, "
            + f"{len(self.train_data)} train-, {len(self.val_data)} val-, and {len(self.test_data)} test samples."
        )

    def get_transform(
        self, pre_model_transforms: list = [], pre_dataset_transforms: list = [], pre_training_transforms: list = []
    ) -> None:
        def build_transforms(transform):
            transform_class, transform_attr = list(transform.items())[0]
            transform_attr = {} if transform_attr is None else transform_attr
            if not transform_class.startswith("torchvision"):
                return import_from_cfg(transform_class)(**transform_attr).get()
            else:
                return import_from_cfg(transform_class)(**transform_attr)

        self.log.debug(f"Building transforms...")

        post_model_transforms = [build_transforms(transform) for transform in pre_model_transforms]
        post_dataset_transforms = [build_transforms(transform) for transform in pre_dataset_transforms]
        post_training_transforms = [build_transforms(transform) for transform in pre_training_transforms]

        self.plain_transform = transforms.Compose(post_model_transforms + post_dataset_transforms)
        self.log.debug(f"Plain transforms: {self.plain_transform}")

        RESIZE_TRANSFORMS = (transforms.Resize, transforms.RandomResizedCrop, transforms.CenterCrop)  # Order matters.
        if any([isinstance(transform, RESIZE_TRANSFORMS) for transform in post_training_transforms]) and any(
            _model_resize_idx := [isinstance(transform, RESIZE_TRANSFORMS) for transform in post_model_transforms]
        ):
            _model_resize_idx.reverse()
            input_size = post_model_transforms[_model_resize_idx.index(True)].size
            for transform in post_training_transforms:
                if isinstance(transform, RESIZE_TRANSFORMS):
                    transform.size = input_size

        self.training_transform = transforms.Compose(post_training_transforms + post_dataset_transforms)
        self.log.debug(f"Training transforms: {self.training_transform}")

    def get_dataset(self, dataset_cfg: DictConfig) -> None:
        self.log.debug(f"Building dataset...")

        self.n_classes = dataset_cfg.n_classes
        self._init_dataset(dataset_cfg)
        self.log.debug(f"Dataset contains {len(self.train_data)} train & {len(self.test_data)} test samples")
        self._make_tailed_dataset(dataset_cfg)
        self._make_open_dataset(dataset_cfg)
        self._limit_dataset()

        self.train_plain_data = deepcopy(self.train_data)
        self.train_plain_data.transform = self.plain_transform
        assert len(self.train_data) == len(self.train_plain_data)

    def _init_dataset(self, dataset_cfg: DictConfig):
        if dataset_cfg.get("data"):
            data = instantiate(dataset_cfg.data, transform=self.plain_transform)
            if dataset_cfg.get("split") and dataset_cfg.split.get("test"):
                self.train_data, self.test_data = split_dataset_class_balanced(
                    data, seed=self.cfg.seed, **dataset_cfg.split.test
                )
            else:
                self.train_data, self.test_data = split_dataset_class_balanced(
                    data, seed=self.cfg.seed, n_samples=0, minimum=0
                )
            self.train_data.transform = self.training_transform
            if dataset_cfg.get("split") and dataset_cfg.split.get("val"):
                self.train_data, self.val_data = split_dataset_class_balanced(
                    self.train_data, seed=self.cfg.seed, **dataset_cfg.split.val
                )
            else:
                self.train_data, self.val_data = split_dataset_class_balanced(
                    data, seed=self.cfg.seed, n_samples=0, minimum=0
                )
            if dataset_cfg.get("split") and dataset_cfg.split.get("osr"):  # TODO: Only works on iNaturalist2018_Partial
                assert dataset_cfg.superclasses
                osr_sclasses = set(dataset_cfg.superclasses) - set(dataset_cfg.data.families)
                self.log.info(f"Loading OSR samples with {dataset_cfg.split.osr.n_samples} samples per superclass.")
                self.log.debug(f"OSR Superclasses: {', '.join(osr_sclasses)}")
                osr_datasets = []
                for sclass in osr_sclasses:
                    sclass_data = instantiate(dataset_cfg.data, transform=self.plain_transform, families=[sclass])
                    sclass_data, _ = split_dataset(sclass_data, dataset_cfg.split.osr.n_samples, seed=self.cfg.seed)
                    osr_datasets.append(sclass_data)
                self.open_data = torch.utils.data.ConcatDataset(osr_datasets)

        else:
            self.train_data, self.test_data, self.val_data = (
                instantiate(dataset_cfg.train, transform=self.training_transform),
                instantiate(dataset_cfg.test, transform=self.plain_transform),
                instantiate(dataset_cfg.val, transform=self.plain_transform) if "val" in dataset_cfg else None,
            )
        if hasattr(dataset_cfg, "openset"):
            self.open_data = instantiate(dataset_cfg.openset, transform=self.plain_transform)
            self.log.info(f"Loaded open dataset with {len(self.open_data)} samples.")

    def _limit_dataset(self):
        if self.cfg.limit.train and self.train_data and self.cfg.limit.train < len(self.train_data):
            self.train_data, _ = split_dataset(self.train_data, self.cfg.limit.train, self.cfg.seed)
            self.log.info(f"\tLimited train data to {len(self.train_data)} samples")
        if self.cfg.limit.val and self.val_data and self.cfg.limit.val < len(self.val_data):
            self.val_data, _ = split_dataset(self.val_data, self.cfg.limit.val, self.cfg.seed)
            self.log.info(f"\tLimited val data to {len(self.val_data)} samples")
        if self.cfg.limit.test and self.test_data and self.cfg.limit.test < len(self.test_data):
            self.test_data, _ = split_dataset(self.test_data, self.cfg.limit.test, self.cfg.seed)
            self.log.info(f"\tLimited test data to {len(self.test_data)} samples")

    def _make_tailed_dataset(self, dataset_cfg):
        if dataset_cfg.tail and dataset_cfg.tail.val:
            self.train_data = make_LT_dataset(
                self.train_data,
                **OmegaConf.to_container(dataset_cfg.tail, resolve=True),
                logger=self.log,
                seed=self.cfg.seed,
                target_transform=None if not dataset_cfg.tail else target_class_mapping,
            )
        if not self.val_data:
            self.log.warning("There is no validation data. Using first sample as validation data.")
            self.val_data = Subset(self.train_data, [0])
            self.val_data.transform = self.plain_transform
        if dataset_cfg.tail and not dataset_cfg.tail.val:
            self.train_data = make_LT_dataset(
                self.train_data,
                **OmegaConf.to_container(dataset_cfg.tail, resolve=True),
                logger=self.log,
                seed=self.cfg.seed,
            )
        if not "val" in dataset_cfg:
            self.log.info(f"Split train dataset into {len(self.train_data)} train & {len(self.val_data)} val samples")
        else:
            self.log.info(f"Post-tailing: Tailed train dataset contains {len(self.train_data)}.")

    def _make_open_dataset(self, dataset_cfg):
        self.open_classes = []
        self.open_samples = []
        train_classes = np.unique(self.train_data.targets)
        self.class_mapping = {cls: cls for cls in train_classes}
        if dataset_cfg.open:
            self.log.debug(f"Making open dataset...")
            np.random.seed(self.cfg.seed)
            if dataset_cfg.open.type == "class":
                n_open_classes = int(len(train_classes) * dataset_cfg.open.split)
                self.log.info(f"Randomly selecting {n_open_classes} open classes from {len(train_classes)} classes")
                self.open_classes = np.random.choice(train_classes, n_open_classes, replace=False)
                self.open_classes.sort()
                self.log.info(f"Open classes: {self.open_classes}")
                self.open_samples = np.where(np.isin(self.train_data.targets, self.open_classes))[0]
                self.train_data = Subset(
                    self.train_data, np.where(~np.isin(self.train_data.targets, self.open_classes))[0]
                )
            elif dataset_cfg.open.type == "sample":
                n_open_samples = int(len(self.train_data) * dataset_cfg.open.split)
                self.log.info(f"Randomly selecting {n_open_samples} open samples from {len(self.train_data)} samples")
                self.open_samples = np.random.choice(len(self.train_data), n_open_samples, replace=False)
                self.train_data = Subset(
                    self.train_data, np.where(~np.isin(np.arange(len(self.train_data)), self.open_samples))[0]
                )
                self.open_classes = list(set(train_classes) - set(np.unique(self.train_data.targets)))
            else:
                raise ValueError(f"Unknown open type: {dataset_cfg.open.type}")

            self.open_classes = list(self.open_classes)
            for open_class in self.open_classes:
                self.class_mapping[open_class] = -1
            if not dataset_cfg.open.val:
                self.val_data = Subset(self.val_data, np.where(~np.isin(self.val_data.targets, self.open_classes))[0])
            if not dataset_cfg.open.test:
                self.test_data = Subset(
                    self.test_data, np.where(~np.isin(self.test_data.targets, self.open_classes))[0]
                )

            if self.cfg.open.reorder:
                self._reorder_classes()
            else:
                self._class_reordered = False
            self.log.info(f"Post-opening: Training dataset contains {len(self.train_data)} samples")

    def _reorder_classes(self, ignore_classes: list = None):
        self._class_reordered = True
        self.log.info("Reordering open classes")
        train_classes = np.unique(self.train_data.targets)
        if not hasattr(self, "class_mapping"):
            self.class_mapping = {cls: cls for cls in train_classes}
        self._n_classes = len(train_classes)
        for new_target, old_target in enumerate(train_classes):
            if new_target == old_target:
                continue
            self.class_mapping[old_target] = new_target
        for dataset in [self.train_data, self.val_data, self.test_data]:
            if isinstance(dataset, Subset):
                dataset.dataset.target_transform = target_class_mapping(self.class_mapping).get()
            elif isinstance(dataset, torch.utils.data.Dataset):
                dataset.target_transform = target_class_mapping(self.class_mapping).get()
            elif dataset is None or isinstance(dataset, str):
                continue
            else:
                raise ValueError(f"Unknown dataset type: {type(dataset)}")

    def get_loaders(self, dataloader_cfg: DictConfig) -> None:
        def _get_loader(loader_cfg, dataset):
            sampler = instantiate(loader_cfg.sampler, dataset=dataset) if loader_cfg.get("sampler") else None
            if sampler and loader_cfg.shuffle:
                self.log.warning("Sampler and shuffle are both set to True. Sampler will be used.")
                loader_cfg.shuffle = False
            return instantiate(loader_cfg, dataset=dataset, sampler=sampler)

        if isinstance(self.train_data, Subset) and self.cfg.dataset.tail:
            self.log.info(
                "Overwritting config to not shuffle train (plain) data, as it is already shuffled during set creation."
            )
            dataloader_cfg.train_loader.shuffle = False
            dataloader_cfg.train_plain_loader.shuffle = False
        elif isinstance(self.train_data, Subset) and dataloader_cfg.train_loader.shuffle:  # None LT'ed dataset
            dataloader_cfg.train_loader.shuffle = False
            dataloader_cfg.train_plain_loader.shuffle = False

            np.random.seed(self.cfg.seed)
            self.train_data.indices = np.random.permutation(self.train_data.indices)
            self.train_plain_data.indices = self.train_data.indices

        if hasattr(self, "_n_classes") and self._n_classes != self.n_classes:
            for loader in [dataloader_cfg.train_loader, dataloader_cfg.train_plain_loader, dataloader_cfg.val_loader]:
                if "n_classes" in loader:
                    loader.n_classes = self._n_classes
        self.train_loader = _get_loader(dataloader_cfg.train_loader, dataset=self.train_data)
        self.train_plain_loader = _get_loader(dataloader_cfg.train_plain_loader, dataset=self.train_plain_data)
        self.val_loader = _get_loader(dataloader_cfg.val_loader, dataset=self.val_data)
        self.test_loader = _get_loader(dataloader_cfg.test_loader, dataset=self.test_data)
        if hasattr(self, "open_data"):
            self.open_loader = _get_loader(dataloader_cfg.test_loader, dataset=self.open_data)

    def get_dataset_info(self, info: bool = True):
        assert self.train_data, "self.train_data is None. Did you call self.get_loaders()?"
        classes = np.unique(self.test_data.targets)
        self.classes = list(set(classes) - {-1})
        self.train_classes, self.class_counts = np.unique(self.train_data.targets, return_counts=True)
        self.n_classes = len(self.classes)
        self.n_open_classes = len(self.open_classes)

        if info:
            class_coverage(self.train_data.targets, self.n_classes, thresholds=[5, 10])
            self.log.info(
                f"Classes: {self.n_classes}, "
                + f"Closed classes: {self.n_classes - self.n_open_classes}, "
                + f"Open classes: {self.n_open_classes} "
                + f"({self.n_open_classes/(self.n_classes if self.n_classes else -1):.2%})"
            )
        self.write_counts_to_file()

    def write_counts_to_file(self):
        try:
            path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "results.txt")
        except:
            self.log.error("Could not find hydra output directory. Writing to results.txt in current directory.")
            path = "results.txt"
        with open(path, "a") as f:
            f.write(
                f"Dataset: {self.dataset_name}, "
                + f"Open_classes: {self.open_classes} ({self.n_open_classes/(self.n_open_classes+self.n_classes):.2%})\n\n"
            )
            f.write(f"Classes {' '.join([str(i) for i in self.train_classes])}\n")
            f.write(f"Counts {' '.join([str(i) for i in self.class_counts])}\n")
