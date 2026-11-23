"""DenseNet dense block: n layers of BN-ReLU-1x1(4g)-BN-ReLU-3x3(g), each concatenated to its input.
Origin: legacy `layers/densenet.py` (a copy of the Keras DenseNet with configurable block counts)."""
import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    def __init__(self, cin, n_layers, growth=32):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            c = cin + i * growth
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(c), nn.ReLU(inplace=True), nn.Conv2d(c, 4 * growth, 1, bias=False),
                nn.BatchNorm2d(4 * growth), nn.ReLU(inplace=True), nn.Conv2d(4 * growth, growth, 3, padding=1, bias=False)))
        self.out_channels = cin + n_layers * growth

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], 1)
        return x
