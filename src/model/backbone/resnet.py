'''
Modified ResNet in PyTorch.

Modifications
[1] input channel: 3 -> 4
[2] Group-Normalization
[3] Learnable Spatial Embedding (no global average pooling)

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import BaseBackbone
from src.model.model_utils import init_normalization

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, norm_type):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = init_normalization(channels=planes, norm_type=norm_type) or nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm2 = init_normalization(channels=planes, norm_type=norm_type) or nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                init_normalization(channels=self.expansion*planes, norm_type=norm_type) # type: ignore
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x))) # type: ignore
        out = self.norm2(self.conv2(out)) # type: ignore
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, norm_type):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.norm1 = init_normalization(channels=planes, norm_type=norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.norm2 = init_normalization(channels=planes, norm_type=norm_type)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.norm3 = init_normalization(channels=self.expansion*planes, norm_type=norm_type)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                init_normalization(channels=self.expansion*planes, norm_type=norm_type)
            )

    def forward(self, x):
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class ResNet(BaseBackbone):
    name = "resnet"
    def __init__(self, in_shape, action_size, norm_type="gn", net_type="resnet50", width_multiplier=1.0):
        super().__init__(in_shape, action_size)

        assert net_type in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        if net_type == "resnet18":
            block = BasicBlock
            num_blocks = [2, 2, 2, 2]
        elif net_type == "resnet34":
            block = BasicBlock
            num_blocks = [3, 4, 6, 3]
        elif net_type == "resnet50":
            block = Bottleneck
            num_blocks = [3, 4, 6, 3]
        elif net_type == "resnet101":
            block = Bottleneck
            num_blocks = [3, 4, 23, 3]
        elif net_type == "resnet152":
            block = Bottleneck
            num_blocks = [3, 8, 36, 3]
        
        f, c, _, _ = in_shape
        in_channel = f * c

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = init_normalization(channels=64, norm_type=norm_type)
        self.stem = nn.Sequential(self.conv1, self.norm1, nn.ReLU(inplace=True))

        self.in_planes = 64
        self.layer1 = self._make_layer(block, num_blocks[0], int(64*width_multiplier), stride=1, norm_type=norm_type)
        self.layer2 = self._make_layer(block, num_blocks[1], int(128*width_multiplier), stride=2, norm_type=norm_type)
        self.layer3 = self._make_layer(block, num_blocks[2], int(256*width_multiplier), stride=2, norm_type=norm_type)
        self.layer4 = self._make_layer(block, num_blocks[3], int(512*width_multiplier), stride=2, norm_type=norm_type)

    def _make_layer(self, block, num_blocks, planes, stride, norm_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        t = x.shape[1]  # (n, t, f, c, h, w)
        x = rearrange(x, 'n t f c h w -> (n t) (f c) h w')
        out = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        x = rearrange(x, '(n t) c h w -> n t c h w', t=t)
        return x, {}

        