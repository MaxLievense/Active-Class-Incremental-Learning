from torch.nn import MSELoss as torch_MSELoss


class MSELoss(torch_MSELoss):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def post_forward(self, logits):
        return logits
