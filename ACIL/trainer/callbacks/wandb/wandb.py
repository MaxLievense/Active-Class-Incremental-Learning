import wandb

from ACIL.trainer.callbacks.base import BaseCallback


class Wandb(BaseCallback):
    def __call__(self, trainer, loss, accuracy, epoch, outputs, data, lr):
        wandb.log({"Train/Loss": loss, "Train/Accuracy": accuracy, "Train/Epoch": epoch, "Train/LR": lr})
