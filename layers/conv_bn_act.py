"""Conv2d -> BatchNorm -> SiLU. Stride / kernel may be (freq, time) tuples.
Origin: research session."""

import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act=True):
        # k = kernel size, s = stride; pad by half the kernel, per axis when k is a tuple
        if isinstance(k, int):
            padding = k // 2
        else:
            padding = (k[0] // 2, k[1] // 2)
        if act:
            activation = nn.SiLU(inplace=True)
        else:
            activation = nn.Identity()
        super().__init__(
            nn.Conv2d(cin, cout, k, s, padding, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            activation,
        )
