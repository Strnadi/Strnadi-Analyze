"""log + per-frequency median subtraction over time (removes stationary noise and device EQ).

Known weakness (base-session audit): when a song fills much of the window the median itself
contains song energy, so 36-64 % of dialect clips lose > 6 dB in some bin (worst for BlBh).
`LogPercentileNorm` is the fix to ablate.
The median is expressed with TopK so it exports. Origin: research session default.
"""

import torch
import torch.nn as nn


class LogMedianNorm(nn.Module):
    out_channels = 1

    def __init__(self, scale=4.0, eps=1e-8):
        super().__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, bands):
        """Band power (B, F, T) -> (B, 1, F, T)."""
        log_bands = torch.log(bands + self.eps)  # (B, F, T)
        n_frames = log_bands.shape[-1]
        # the median over time is the k-th smallest value of every bin
        k = (n_frames - 1) // 2 + 1
        k_smallest = torch.topk(log_bands, k, dim=-1, largest=False).values  # (B, F, k), ascending
        median = k_smallest[..., -1:]  # (B, F, 1): the largest of the k smallest
        normalised = (log_bands - median) / self.scale
        return normalised.unsqueeze(1)  # (B, 1, F, T)
