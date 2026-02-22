from abc import ABCMeta
import torch.nn as nn

class BaseNeck(nn.Module, metaclass=ABCMeta):
    name = "base_neck"
    def __init__(self, in_shape, action_size):
        super().__init__()
        self.in_shape = in_shape
        self.action_size = action_size
    
    @classmethod
    def get_name(cls):
        return cls.name
    
    def reset_parameters(self, **kwargs):
        for _, layer in self.named_children():
            modules = [m for m in layer.children()]
            for m in modules:
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()