"""Conv2d -> BatchNorm -> SiLU. Stride / kernel may be (freq, time) tuples. Origin: research session."""
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act=True):
        pad = (k // 2) if isinstance(k, int) else (k[0] // 2, k[1] // 2)
        super().__init__(nn.Conv2d(cin, cout, k, s, pad, groups=groups, bias=False), nn.BatchNorm2d(cout),
                         nn.SiLU(inplace=True) if act else nn.Identity())
