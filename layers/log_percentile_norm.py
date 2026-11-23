"""log + per-frequency low-percentile noise-floor subtraction.

Same idea as `LogMedianNorm`, but the floor is the q-th percentile (default 15 %) over time, which stays on the
noise even when the song occupies most of the window, so song energy is not subtracted from itself.
Origin: base-session review (audit of the median front-end).
"""
import math
import torch
import torch.nn as nn


class LogPercentileNorm(nn.Module):
    out_channels = 1

    def __init__(self, q=0.15, scale=4.0, eps=1e-8):
        super().__init__(); self.q, self.scale, self.eps = q, scale, eps

    def forward(self, bands):
        y = torch.log(bands + self.eps)
        k = max(1, math.ceil(self.q * y.shape[-1]))
        floor = torch.topk(y, k, dim=-1, largest=False).values[..., -1:]
        return ((y - floor) / self.scale).unsqueeze(1)
