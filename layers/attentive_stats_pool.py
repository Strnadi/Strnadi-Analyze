"""Attentive statistics pooling over time: learned frame weights -> weighted mean and std.

(B, D, T) -> pooled (B, 2D) and the frame weights (B, T), which double as a "where is the song" signal.
Origin: legacy `AttentiveStatsPool1D` (SimCLR CRNN) and research StrnadNet.
"""
import torch
import torch.nn as nn


class AttentiveStatsPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.att = nn.Conv1d(d, 1, 1)

    def forward(self, h):
        a = torch.softmax(self.att(h), dim=-1)                       # (B, 1, T)
        mu = (h * a).sum(-1)
        sd = ((h - mu.unsqueeze(-1)) ** 2 * a).sum(-1).clamp_min(1e-6).sqrt()
        return torch.cat([mu, sd], 1), a.squeeze(1)
