"""log + per-frequency low-percentile noise-floor subtraction.

Same idea as `LogMedianNorm`, but the floor is the q-th percentile (default 15 %) over time, which
stays on the noise even when the song occupies most of the window, so song energy is not subtracted
from itself.
Origin: base-session review (audit of the median front-end).
"""

import math

import torch
import torch.nn as nn


class LogPercentileNorm(nn.Module):
    out_channels = 1

    def __init__(self, q=0.15, scale=4.0, eps=1e-8):
        super().__init__()
        self.q = q
        self.scale = scale
        self.eps = eps

    def forward(self, bands):
        """Band power (B, F, T) -> (B, 1, F, T)."""
        log_bands = torch.log(bands + self.eps)  # (B, F, T)
        n_frames = log_bands.shape[-1]
        # the q-th percentile over time is the k-th smallest value of every bin
        k = max(1, math.ceil(self.q * n_frames))
        k_smallest = torch.topk(log_bands, k, dim=-1, largest=False).values  # (B, F, k), ascending
        noise_floor = k_smallest[..., -1:]  # (B, F, 1): the largest of the k smallest
        normalised = (log_bands - noise_floor) / self.scale
        return normalised.unsqueeze(1)  # (B, 1, F, T)
