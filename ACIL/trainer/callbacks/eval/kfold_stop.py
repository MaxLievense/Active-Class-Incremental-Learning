from ACIL.trainer.callbacks.eval.stop import EvalStop


class KFoldStop(EvalStop):
    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.reset()
        self.patience = int(self.cfg.patience * self.cfg.n_folds)
        self.warmup = int(self.cfg.warmup * self.cfg.n_folds)
        self.log.info(f"Initalized {self.cfg.n_folds}-FoldStop with patience {self.patience} and warmup {self.warmup}.")

    def __call__(self, trainer, epoch, **_):
        res = super().__call__(trainer, epoch)
        if res:
            trainer.model._n_fold = 0
        return res
