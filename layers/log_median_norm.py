"""log + per-frequency median subtraction over time (removes stationary noise and device EQ).

Known weakness (base-session audit): when a song fills much of the window the median itself contains song energy,
so 36-64 % of dialect clips lose > 6 dB in some bin (worst for BlBh). `LogPercentileNorm` is the fix to ablate.
The median is expressed with TopK so it exports. Origin: research session default.
"""
import torch
import torch.nn as nn


class LogMedianNorm(nn.Module):
    out_channels = 1

    def __init__(self, scale=4.0, eps=1e-8):
        super().__init__(); self.scale, self.eps = scale, eps

    def forward(self, bands):  # (B, F, T) power -> (B, 1, F, T)
        y = torch.log(bands + self.eps)
        k = (y.shape[-1] - 1) // 2 + 1
        med = torch.topk(y, k, dim=-1, largest=False).values[..., -1:]
        return ((y - med) / self.scale).unsqueeze(1)
