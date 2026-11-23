"""CBAM spatial attention: a 7x7 conv over [channel-mean, channel-max] gives a sigmoid mask over (freq, time).
Origin: legacy `model/autoencoder.py::SpatialAttention`."""
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        m = torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * m
