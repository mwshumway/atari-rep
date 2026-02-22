import torch.nn as nn
from abc import ABCMeta, abstractmethod

class BaseBackbone(nn.Module, metaclass=ABCMeta):
    name = "base_backbone"
    def __init__(self, in_shape, action_size):
        super().__init__()
        self.in_shape = in_shape
        self.action_size = action_size
    
    @classmethod
    def get_name(cls):
        return cls.name
    
    @abstractmethod
    def forward(self, x):
        """
        :param x (torch.Tensor): (n, t, c, h, w)
        :return x (torch.Tensor): (n, t, d)
        """
        pass

    def reset_parameters(self):
        for _, layer in self.named_children():
            modules = [m for m in layer.children()]
            for m in modules:
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()