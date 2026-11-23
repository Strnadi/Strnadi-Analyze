"""Attentive statistics pooling over time: learned frame weights -> weighted mean and std.

(B, D, T) -> pooled (B, 2D) and the frame weights (B, T), which double as a "where is the song"
signal.
Origin: legacy `AttentiveStatsPool1D` (SimCLR CRNN) and research StrnadNet.
"""

import torch
import torch.nn as nn


class AttentiveStatsPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.att = nn.Conv1d(d, 1, 1)  # one attention score per frame

    def forward(self, frames):
        """Frame series (B, D, T) -> pooled (B, 2D), frame weights (B, T)."""
        attention_weights = torch.softmax(self.att(frames), dim=-1)  # (B, 1, T), sums to 1 over T
        mean = (frames * attention_weights).sum(-1)  # (B, D)
        squared_deviation = (frames - mean.unsqueeze(-1)) ** 2  # (B, D, T)
        variance = (squared_deviation * attention_weights).sum(-1)  # (B, D)
        std = variance.clamp_min(1e-6).sqrt()
        pooled = torch.cat([mean, std], 1)  # (B, 2D)
        return pooled, attention_weights.squeeze(1)
