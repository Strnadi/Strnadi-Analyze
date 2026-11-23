"""DenseNet dense block: n layers of BN-ReLU-1x1(4g)-BN-ReLU-3x3(g), each concatenated to its input.
Origin: legacy `layers/densenet.py` (a copy of the Keras DenseNet with configurable block
counts)."""

import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, cin, n_layers, growth=32):
        super().__init__()
        bottleneck_channels = 4 * growth
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # every layer sees the block input plus the outputs of all earlier layers
            layer_in_channels = cin + i * growth
            layer = nn.Sequential(
                nn.BatchNorm2d(layer_in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(layer_in_channels, bottleneck_channels, 1, bias=False),
                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_channels, growth, 3, padding=1, bias=False),
            )
            self.layers.append(layer)
        self.out_channels = cin + n_layers * growth

    def forward(self, x):
        """(B, cin, F, T) -> (B, cin + n_layers * growth, F, T)."""
        for layer in self.layers:
            new_features = layer(x)  # (B, growth, F, T)
            x = torch.cat([x, new_features], 1)
        return x
