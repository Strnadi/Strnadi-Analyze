"""log + per-example mean/std standardisation (the plain baseline normalisation).
Origin: legacy `model/mobilenet.py::mean_std_normalization`; research `norm='mean'`."""

import torch
import torch.nn as nn


class LogMeanStdNorm(nn.Module):
    out_channels = 1

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, bands):
        """Band power (B, F, T) -> (B, 1, F, T)."""
        log_bands = torch.log(bands + self.eps)  # (B, F, T)
        # one mean and one std per example, over all bins and frames
        mean = log_bands.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        std = log_bands.std(dim=(1, 2), keepdim=True) + 1e-5  # (B, 1, 1)
        standardised = (log_bands - mean) / std
        return standardised.unsqueeze(1)  # (B, 1, F, T)
