import numpy as np


class EnvState(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.asarray(*args, **kwargs).view(cls)


class Perception(EnvState):
    pass
