from torch.nn import CrossEntropyLoss


class CrossEntropy(CrossEntropyLoss):
    def __init__(self, model, **cfg):
        super().__init__(**cfg)

    def post_forward(self, logits):
        return logits
