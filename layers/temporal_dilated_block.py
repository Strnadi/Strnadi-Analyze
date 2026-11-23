"""Residual dilated depthwise temporal conv block (ConvNeXt-style, 1-D). Three of them with dilation
1/2/4 give ~1.2 s of receptive field at 40 ms frames: enough to see the ORDER of neighbouring
elements.
Origin: research session (StrnadNet temporal module)."""

import torch.nn as nn


class TemporalDilatedBlock(nn.Module):
    def __init__(self, d, k=5, dil=1, drop=0.1):
        super().__init__()
        # depthwise conv along time; this padding keeps T frames for an odd kernel size k
        padding = dil * (k // 2)
        self.conv = nn.Conv1d(d, d, k, padding=padding, dilation=dil, groups=d)
        self.norm = nn.BatchNorm1d(d)
        # pointwise (1x1) convs: expand to 2d channels, then back to d
        self.pw = nn.Sequential(
            nn.Conv1d(d, 2 * d, 1),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Conv1d(2 * d, d, 1),
        )

    def forward(self, x):
        """Frame series (B, D, T) -> (B, D, T)."""
        residual = self.pw(self.norm(self.conv(x)))
        return x + residual
