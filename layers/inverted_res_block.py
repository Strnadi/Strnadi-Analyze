"""Inverted-residual (expand 1x1 -> depthwise 3x3 -> project 1x1) variant of `ResBlock`, with the
same pooled-projection skip. Mobile-friendly: far fewer MACs per channel.
Origin: research session (`dw=True`)."""

import torch.nn as nn

from .conv_bn_act import ConvBNAct


class InvertedResBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1), expand=3):
        super().__init__()
        stride = tuple(stride)  # (freq, time)
        expanded_channels = cin * expand
        self.body = nn.Sequential(
            ConvBNAct(cin, expanded_channels, 1),  # expand
            ConvBNAct(expanded_channels, expanded_channels, 3, stride, groups=expanded_channels),
            ConvBNAct(expanded_channels, cout, 1, act=False),  # project
        )

        keeps_shape = cin == cout and stride == (1, 1)
        if keeps_shape:
            self.skip = nn.Identity()
        else:
            # the skip must follow the body: average-pool to the new resolution, 1x1 conv to cout
            if stride != (1, 1):
                skip_pool = nn.AvgPool2d(stride, stride, ceil_mode=True)
            else:
                skip_pool = nn.Identity()
            self.skip = nn.Sequential(
                skip_pool,
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """(B, cin, F, T) -> (B, cout, F', T') with F', T' set by `stride`."""
        return self.act(self.body(x) + self.skip(x))
