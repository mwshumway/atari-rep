import torch.nn as nn
from src.models.backbones.base import BaseBackbone
from src.models.layers import init_normalization


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = init_normalization(channels=channels, norm_type='gn')
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = init_normalization(channels=channels, norm_type='gn')
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out) # type: ignore
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out) # type: ignore
        out += residual
        out = self.relu(out)
        return out


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = init_normalization(channels=out_channels, norm_type='gn')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)  # type: ignore
        x = self.relu(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class Impala(BaseBackbone):
    name = 'impala'
    def __init__(self, in_shape, action_size, **kwargs):
        super().__init__(in_shape, action_size)
        in_channels = in_shape[0] * in_shape[1]  # f * c
        self.blocks = nn.Sequential(
            ImpalaBlock(in_channels, 32),
            ImpalaBlock(32, 64),
            ImpalaBlock(64, 64),
        )
    
    def forward(self, x):
        n, t, f, c, h, w = x.shape
        x = x.reshape(n * t, f * c, h, w) # (n * t, f * c, h, w)
        x = self.blocks(x) # (n * t, 64, h', w')
        x = x.reshape(n, t, 64, x.shape[-2], x.shape[-1])  # (n, t, 64, h', w')
        info = {}
        return x, info
