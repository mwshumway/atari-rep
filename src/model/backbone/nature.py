from einops import rearrange
from torch import nn

from .base import BaseBackbone

class Nature(BaseBackbone):
    name = 'nature'
    def __init__(self,
                 in_shape,
                 action_size):
        super().__init__(in_shape, action_size)
        f, c, h, w = in_shape
        in_channels = f * c

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), 
            nn.ReLU(),
        )
                
    def forward(self, x):
        n, t, f, c, h, w = x.shape
        x = rearrange(x, 'n t f c h w -> (n t) (f c) h w')
        x = self.layers(x)
        x = rearrange(x, '(n t) c h w -> n t c h w', t=t)
        info = {}
            
        return x, info