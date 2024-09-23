from pprint import pprint

import torch

from src.utils.metrics import class_metrics


def test_class_metrics_closed_acc():
    y_true = torch.tensor([0, 1, 2, 0])
    output = torch.tensor(
        [
            [0.9, 0.05, 0.05],  # 0
            [0.05, 0.9, 0.05],  # 1
            [0.05, 0.05, 0.9],  # 2
            [0.9, 0.05, 0.05],  # 0
        ]
    )
    output = class_metrics(y_true, output)
    pprint(output)
    assert output[0].accuracy == 1.0
    assert output[1] == None


def test_class_metrics_closed():
    y_true = torch.tensor([0, 1, 2, 0, 0, 1, 2, 0])
    output = torch.tensor(
        [
            [0.9, 0.05, 0.05],  # 0
            [0.05, 0.9, 0.05],  # 1
            [0.05, 0.05, 0.9],  # 2
            [0.05, 0.05, 0.9],  # 2 <
            [0.9, 0.05, 0.05],  # 0
            [0.05, 0.9, 0.05],  # 1
            [0.05, 0.05, 0.9],  # 2
            [0.05, 0.05, 0.9],  # 2 <
        ]
    )
    output = class_metrics(y_true, output)
    pprint(output)
    assert output[0].accuracy == 6 / 8
    assert output[1] == None
    assert output[0].recall == 7.5 / 9  # 0.5 * 3 + 3 + 3


def test_class_metrics_open():
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, -1, -1, -1])
    output = torch.tensor(
        [
            [0.9, 0.05, 0.05],  # 0
            [0.05, 0.9, 0.05],  # 1
            [0.05, 0.05, 0.9],  # 2
            [0.9, 0.05, 0.05],  # 0
            [0.05, 0.9, 0.05],  # 1
            [0.05, 0.05, 0.9],  # 2
            [0.05, 0.0, 0.01],  # -1
            [0.04, 0.0, 0.01],  # -1
            [0.05, 0.9, 0.01],  # 1 <
        ]
    )
    output = class_metrics(y_true, output)
    pprint(output)
    assert output[0].accuracy == 1
    assert output[0].recall == 1
    assert output[1].accuracy == 8 / 9
    assert output[1].per_class_accuracy[0] == 2 / 3
