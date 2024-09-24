from ACIL.utils.base import Base


class Callbacks(Base):
    def __init__(self, **cfg: dict) -> None:
        super().__init__({})
        self.log.debug("Building callbacks...")
        self.callbacks = cfg
        self.log.info("Loaded callbacks.")

    def __call__(self, trainer, *args, **kwargs):
        """If a callback returns True, the trainer will terminate."""
        return any(callback(trainer, *args, **kwargs) for callback in self.callbacks.values())
