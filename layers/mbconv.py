"""MobileNetV2 MBConv block (ReLU6; identity skip only when stride 1 and channels match).
Origin: legacy `model/autoencoder.py::mbconv_block`."""
import torch.nn as nn


class MBConv(nn.Module):
    def __init__(self, cin, cout, expand=6, stride=1, k=3):
        super().__init__()
        mid = cin * expand; layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(mid, mid, k, stride, k // 2, groups=mid, bias=False), nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
                   nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.body = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.residual else self.body(x)
