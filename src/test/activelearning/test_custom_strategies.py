from pytest_unordered import unordered
from torch import Tensor

N_SAMPLES = 5
LOGITS = Tensor(
    [
        [0.12, 0.2, 0.3, 0.4, 0.5],  # 0
        [0.2, 0.3, 0.4, 0.5, 0.6],  # 1
        [1.0, 0.0, 0.0, 0.0, 0.0],  # 2
        [0.7, 0.8, 0.0, 0.0, 0.91],  # 3
        [0.9, 0.3, 0.0, 0.0, 0.0],  # 4
        [0.0, 0.1, 0.0, 0.0, 0.0],  # 5
        [0.2, 0.0, 0.0, 0.0, 0.11],  # 6
        [0.0, 0.0, 0.0, 0.0, 0.0],  # 7
    ]
)


def test_uncertainty():
    from src.data.query.strategy.uncertainty import Uncertainty

    strategy = Uncertainty(None)
    scores = strategy.score(LOGITS)
    query = strategy.select({idx: score for idx, score in enumerate(scores)}, N_SAMPLES)
    assert query == [7, 5, 6, 0, 1]


def test_margin():
    from src.data.query.strategy.margin import Margin

    strategy = Margin(None)
    scores = strategy.score(LOGITS)
    query = strategy.select({idx: score for idx, score in enumerate(scores)}, N_SAMPLES)
    assert query == unordered([7, 6, 0, 1, 5])
