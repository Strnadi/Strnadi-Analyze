"""DenseNet transition: BN-ReLU-1x1 (channel compression) -> 2x2 average pool.
(B, cin, F, T) -> (B, int(cin * reduction), F // 2, T // 2).
Origin: legacy `layers/densenet.py::transition_block`."""

import torch.nn as nn


class DenseTransition(nn.Sequential):
    def __init__(self, cin, reduction=0.5):
        self.out_channels = int(cin * reduction)
        super().__init__(
            nn.BatchNorm2d(cin),
            nn.ReLU(inplace=True),
            nn.Conv2d(cin, self.out_channels, 1, bias=False),
            nn.AvgPool2d(2, 2),
        )
