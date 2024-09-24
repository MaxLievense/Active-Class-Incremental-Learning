from torch.nn import CrossEntropyLoss


class CrossEntropy(CrossEntropyLoss):
    def post_forward(self, logits):
        return logits
