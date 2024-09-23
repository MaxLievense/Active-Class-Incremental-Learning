from src.trainer.callbacks.base import BaseCallback


class Checkpoint(BaseCallback):
    def __call__(self, trainer, epoch, **_):
        if epoch and epoch % self.cfg.every_n_epochs == 0:
            trainer.model.save(f"epoch_{epoch}")
            self.log.debug(f"Saved checkpoint at epoch {epoch}.")
