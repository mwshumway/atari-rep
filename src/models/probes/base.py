from abc import ABCMeta
import torch.nn as nn


class BaseProbe(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        pass