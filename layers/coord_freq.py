"""Append a frequency-coordinate channel (-1 .. 1 along F), so convolutions can tell absolute pitch
(CoordConv along frequency). Origin: research session (StrnadNet input)."""

import torch
import torch.nn as nn


class CoordFreq(nn.Module):
    def forward(self, x):
        """(B, C, F, T) -> (B, C + 1, F, T); the new last channel holds each bin's position."""
        B, _, n_bins, T = x.shape
        bin_positions = torch.linspace(-1, 1, n_bins, device=x.device, dtype=x.dtype)  # (F,)
        coord_channel = bin_positions.view(1, 1, n_bins, 1)  # (1, 1, F, 1)
        coord_channel = coord_channel.expand(B, 1, n_bins, T)  # (B, 1, F, T)
        return torch.cat([x, coord_channel], 1)
