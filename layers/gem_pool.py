"""Generalised-mean pooling over (freq, time) with a learnable exponent: (B, C, F, T) -> (B, C).
Origin: legacy `layers/global_gem2d.py`."""
import torch
import torch.nn as nn


class GeMPool2D(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p))); self.eps = eps

    def forward(self, x):
        return x.clamp_min(self.eps).pow(self.p).mean(dim=(2, 3)).pow(1.0 / self.p)
