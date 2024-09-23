import numpy as np
import pytest
import torch
from hydra.utils import instantiate

import hydra

DATASETS = {
    # "Places365": ["_LT_open20classes"],
    # "tiny-imagenet-200": ["_LT_open20classes"],
    "MNIST": ["_LT_open20classes", "_XLT_open70samples"],
    "CIFAR100": ["_LT_open20classes", "_XLT_open70samples"],
}


@pytest.fixture(params=[f"{dataset}{ext}" for dataset, exts in DATASETS.items() for ext in exts])
def cfg(request):
    try:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    except:
        pass

    hydra.initialize(version_base="1.2", config_path="../../.")
    return hydra.compose(
        "main.yaml",
        overrides=[
            "+data=normal",
            f"+data@data.dataset=datasets/{request.param}",
            "+data/transforms@model.transforms=32_center",
        ],
    )


def test_dataset_openset(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = instantiate(cfg.data, device=device, _recursive_=False)

    assert len(data.train_data) == len(data.train_plain_data)
    assert (np.unique(data.train_data.targets) == np.unique(data.train_plain_data.targets)).all()
    closed_classes = list(set(data.class_mapping.values()) - {-1})

    for (_, target), (_, target_plain) in zip(data.train_loader, data.train_plain_loader):
        assert np.all([t == p for t, p in zip(target, target_plain)])
        assert np.all([t in closed_classes for t in target])
        break

    for (_, target_val), (_, target_test) in zip(data.val_loader, data.test_loader):
        assert np.all([t in [-1, *closed_classes] for t in target_val])
        assert np.all([t in [-1, *closed_classes] for t in target_test])
        break

    new_class = data.open_classes[0]
    data.discover_class(new_class)
    assert len(closed_classes) < len(set(data.class_mapping.values()))
