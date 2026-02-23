import numpy as np
from abc import ABCMeta


class BaseBuffer(metaclass=ABCMeta):
    name = "base_buffer"

    def __init__(self):
        super().__init__()

    @classmethod
    def get_name(cls):
        return cls.name

    def store(self, obs: np.ndarray, action: int, reward: float, done: bool):
        pass
    
    def sample(self, batch_size: int):
        pass
    
    def save_buffer(self):
        pass

