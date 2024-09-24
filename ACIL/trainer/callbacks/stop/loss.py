import wandb

from ACIL.trainer.callbacks.base import Base


class Loss(Base):
    def __init__(self, **cfg):
        super().__init__(cfg)
        self.reset()

    def reset(self):
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.last_epoch = -1

    def __call__(self, trainer, loss, epoch, **_):
        if epoch < self.last_epoch:
            self.reset()

        self.last_epoch = epoch
        if epoch < self.cfg.warmup:
            return False

        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.cfg.patience:
                self.log.debug(f"Early stopping at epoch {epoch+1}.")
                wandb.log({"Train/Stop_epoch": epoch + 1}, commit=False)
                return True
        return False
