"""Attentive statistics pooling over all (freq, time) cells of a feature map:
(B, C, F, T) -> (B, 2C).

Origin: legacy `layers/attentive_stats_pool.py` (production DenseNet / MobileNet heads). Note it
treats the map as an unordered bag of patches, so element ORDER (BhBl vs BlBh) is not visible to
anything after it.
"""

import torch
import torch.nn as nn


class AttentiveStatsPool2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = nn.Linear(c, 1, bias=False)  # one attention score per cell

    def forward(self, x):
        """(B, C, F, T) -> (B, 2C): attention-weighted mean and std over the F * T cells."""
        cells = x.flatten(2).transpose(1, 2)  # (B, N, C) with N = F * T
        attention_weights = torch.softmax(self.w(cells), dim=1)  # (B, N, 1), sums to 1 over cells
        mean = (cells * attention_weights).sum(1)  # (B, C)
        squared_deviation = (cells - mean.unsqueeze(1)) ** 2  # (B, N, C)
        variance = (squared_deviation * attention_weights).sum(1)  # (B, C)
        std = variance.add(1e-9).sqrt()
        return torch.cat([mean, std], 1)
