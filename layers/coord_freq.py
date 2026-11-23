"""Append a frequency-coordinate channel (-1 .. 1 along F), so convolutions can tell absolute pitch
(CoordConv along frequency). Origin: research session (StrnadNet input)."""
import torch
import torch.nn as nn


class CoordFreq(nn.Module):
    def forward(self, x):  # (B, C, F, T) -> (B, C + 1, F, T)
        B, _, Fq, T = x.shape
        f = torch.linspace(-1, 1, Fq, device=x.device, dtype=x.dtype).view(1, 1, Fq, 1).expand(B, 1, Fq, T)
        return torch.cat([x, f], 1)
