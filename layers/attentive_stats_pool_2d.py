"""Attentive statistics pooling over all (freq, time) cells of a feature map: (B, C, F, T) -> (B, 2C).

Origin: legacy `layers/attentive_stats_pool.py` (production DenseNet / MobileNet heads). Note it treats the map
as an unordered bag of patches, so element ORDER (BhBl vs BlBh) is not visible to anything after it.
"""
import torch
import torch.nn as nn


class AttentiveStatsPool2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = nn.Linear(c, 1, bias=False)

    def forward(self, x):
        h = x.flatten(2).transpose(1, 2)                             # (B, N, C)
        a = torch.softmax(self.w(h), dim=1)                          # (B, N, 1)
        mu = (h * a).sum(1)
        sd = ((h - mu.unsqueeze(1)) ** 2 * a).sum(1).add(1e-9).sqrt()
        return torch.cat([mu, sd], 1)
