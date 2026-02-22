from abc import ABCMeta
import torch.nn as nn


class BaseHead(nn.Module, metaclass=ABCMeta):
    name = "base_head"
    def __init__(self, in_shape, action_size):
        super().__init__()
        self.in_shape = in_shape
        self.action_size = action_size

    @classmethod
    def get_name(cls):
        return cls.name
