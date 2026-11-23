"""Inverted-residual (expand 1x1 -> depthwise 3x3 -> project 1x1) variant of `ResBlock`, with the same
pooled-projection skip. Mobile-friendly: far fewer MACs per channel. Origin: research session (`dw=True`)."""
import torch.nn as nn
from .conv_bn_act import ConvBNAct


class InvertedResBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1), expand=3):
        super().__init__()
        stride = tuple(stride); mid = cin * expand
        self.body = nn.Sequential(ConvBNAct(cin, mid, 1), ConvBNAct(mid, mid, 3, stride, groups=mid),
                                  ConvBNAct(mid, cout, 1, act=False))
        self.skip = nn.Identity() if (cin == cout and stride == (1, 1)) else nn.Sequential(
            nn.AvgPool2d(stride, stride, ceil_mode=True) if stride != (1, 1) else nn.Identity(),
            nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))
