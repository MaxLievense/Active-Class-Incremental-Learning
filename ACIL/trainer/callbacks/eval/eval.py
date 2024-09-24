from ACIL.trainer.callbacks.base import BaseCallback


class Eval(BaseCallback):
    def __call__(self, trainer, epoch, **_):

        if (epoch | self.cfg.at_start) and (epoch + 1) % self.cfg.every_n_epochs == 0:
            trainer.eval(loader=self.cfg.loader)
