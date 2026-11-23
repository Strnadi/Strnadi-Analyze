"""MobileNetV2 MBConv block (ReLU6; identity skip only when stride 1 and channels match).
Origin: legacy `model/autoencoder.py::mbconv_block`."""

import torch.nn as nn


class MBConv(nn.Module):
    def __init__(self, cin, cout, expand=6, stride=1, k=3):
        super().__init__()
        expanded_channels = cin * expand
        layers = []
        if expand != 1:
            # 1x1 expansion (skipped when there is nothing to expand)
            layers += [
                nn.Conv2d(cin, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU6(inplace=True),
            ]
        depthwise = nn.Conv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=k,
            stride=stride,
            padding=k // 2,
            groups=expanded_channels,
            bias=False,
        )
        layers += [
            depthwise,
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU6(inplace=True),
            # 1x1 projection, no activation after it
            nn.Conv2d(expanded_channels, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        ]
        self.body = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        """(B, cin, F, T) -> (B, cout, F', T') with F', T' set by `stride`."""
        out = self.body(x)
        if self.residual:
            return x + out
        return out
