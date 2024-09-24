from ACIL.utils.base import Base


class BaseCallback(Base):
    def __init__(self, **cfg: dict) -> None:
        # TODO: Fix parent init: TypeError('BaseCallback.__init__() takes 1 positional argument but 2 were given')
        super().__init__(cfg)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
