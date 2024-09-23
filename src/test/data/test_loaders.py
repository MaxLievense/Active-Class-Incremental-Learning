import pytest
import torch
from hydra.utils import instantiate
from torchvision import transforms

import hydra

DATASETS = {
    "Places365": ["_LT", "_LT_OLTR", "_XLT"],
    "tiny-imagenet-200": ["_LT", "_XLT"],
    "MNIST": ["_LT", "_XLT"],
    "CIFAR100": ["_LT", "_XLT", "_IM", "_IM_RIDE"],
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
            "+data/transforms@model.transforms=256_center",
            "data.limit.train=10000",
            "data.limit.val=1000",
            "data.limit.test=1000",
            "data.seed=42",
        ],
    )


def test_dataset_loading(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = instantiate(cfg.data, device=device, _recursive_=False)

    data.train_data.transform = transforms.ToTensor()
    data.train_plain_data.transform = transforms.ToTensor()
    data.val_data.transform = transforms.ToTensor()
    data.test_data.transform = transforms.ToTensor()

    assert len(data.train_data) == len(data.train_plain_data)
    assert data.train_data[0][1] == data.train_plain_data[0][1]
    assert data.train_data[0][0].shape == data.train_plain_data[0][0].shape
    assert all([td == pld for td, pld in zip(data.train_data.targets, data.train_plain_data.targets)])
    if isinstance(data.train_data, torch.utils.data.Subset):
        assert isinstance(data.train_plain_data, torch.utils.data.Subset)
        assert all([td == pld for td, pld in zip(data.train_data.indices, data.train_plain_data.indices)])

    # Transforms
    if cfg.data.training_transforms and False:  # disabled
        assert (
            data.train_data.transform
            != data.train_plain_data.transform
            == data.val_data.transform
            == data.test_data.transform
        )

    # Dataloaders
    for (train_data, train_labels), (train_plain_data, train_plain_labels) in zip(
        data.train_loader, data.train_plain_loader
    ):
        assert train_data.shape == train_plain_data.shape
        if isinstance(train_labels, torch.Tensor):
            assert torch.all(train_labels == train_plain_labels)
        else:
            assert train_labels == train_plain_labels
        break
