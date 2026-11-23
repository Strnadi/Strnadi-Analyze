"""CBAM spatial attention: a 7x7 conv over [channel-mean, channel-max] gives a sigmoid mask over
(freq, time). Origin: legacy `model/autoencoder.py::SpatialAttention`."""

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        """(B, C, F, T) -> (B, C, F, T): x times a mask in (0, 1) shared by all channels."""
        channel_mean = x.mean(1, keepdim=True)  # (B, 1, F, T)
        channel_max = x.amax(1, keepdim=True)  # (B, 1, F, T)
        summary = torch.cat([channel_mean, channel_max], 1)  # (B, 2, F, T)
        mask = torch.sigmoid(self.conv(summary))  # (B, 1, F, T)
        return x * mask
