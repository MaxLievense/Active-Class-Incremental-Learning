from torch.nn import MSELoss as torch_MSELoss


class MSELoss(torch_MSELoss):
    def post_forward(self, logits):
        return logits
