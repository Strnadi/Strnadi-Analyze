"""Basic residual block (two 3x3 convs) with an average-pool + 1x1 projection skip; stride is
(freq, time). Origin: research session (StrnadNet body)."""

import torch.nn as nn

from .conv_bn_act import ConvBNAct


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1, 1)):
        super().__init__()
        stride = tuple(stride)  # (freq, time)
        self.body = nn.Sequential(
            ConvBNAct(cin, cout, 3, stride),
            ConvBNAct(cout, cout, 3, act=False),
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
