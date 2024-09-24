from copy import deepcopy

import wandb

from ACIL.trainer.callbacks.base import Base


class EvalStop(Base):
    def __init__(self, **cfg):
        super().__init__(cfg)
        self.reset()
        self.patience = self.cfg.patience
        self.warmup = self.cfg.warmup

    def reset(self):
        self.best_val = -float("inf")
        self.patience_counter = 0
        self.last_epoch = -1

    def __call__(self, trainer, epoch, **_):
        if epoch < self.last_epoch:
            self.reset()

        self.last_epoch = epoch
        if epoch < self.warmup:
            return False

        results = trainer.eval(loader=self.cfg.loader)

        val = getattr(results, self.cfg.metric)
        if val > 0.999:
            self.log.debug(f"Early stopping at epoch {epoch+1} ({self.cfg.metric} (near 100%): {val:.8%}).")
            wandb.log({"Train/Stop_epoch": epoch + 1}, commit=False)
            return True
        if val > self.best_val:
            self.best_val = val
            self.patience_counter = 0
            self.log.debug(f"New best val {self.cfg.metric}: {val:.2%}")
            if self.cfg.checkpoint:
                self.best_dict_state = deepcopy(trainer.model.network.network.state_dict())
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.log.debug(f"Early stopping at epoch {epoch+1} ({self.cfg.metric}: {val:.2%}/{self.best_val:.2%}).")
                wandb.log({"Train/Stop_epoch": epoch + 1}, commit=False)
                if self.cfg.checkpoint and hasattr(self, "best_dict_state"):
                    self.log.info("Loading best model...")
                    trainer.model.network.network.load_state_dict(self.best_dict_state)
                    del self.best_dict_state
                return True
        return False
