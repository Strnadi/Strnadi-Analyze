"""Basic residual block (two 3x3 convs) with an average-pool + 1x1 projection skip; stride is (freq, time).
Origin: research session (StrnadNet body)."""
import torch.nn as nn
from .conv_bn_act import ConvBNAct


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        stride = tuple(stride)
        self.body = nn.Sequential(ConvBNAct(cin, cout, 3, stride), ConvBNAct(cout, cout, 3, act=False))
        self.skip = nn.Identity() if (cin == cout and stride == (1, 1)) else nn.Sequential(
            nn.AvgPool2d(stride, stride, ceil_mode=True) if stride != (1, 1) else nn.Identity(),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))
