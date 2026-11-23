"""log + per-example mean/std standardisation (the plain baseline normalisation).
Origin: legacy `model/mobilenet.py::mean_std_normalization`; research `norm='mean'`."""
import torch
import torch.nn as nn


class LogMeanStdNorm(nn.Module):
    out_channels = 1

    def __init__(self, eps=1e-8):
        super().__init__(); self.eps = eps

    def forward(self, bands):
        y = torch.log(bands + self.eps)
        mu = y.mean(dim=(1, 2), keepdim=True); sd = y.std(dim=(1, 2), keepdim=True) + 1e-5
        return ((y - mu) / sd).unsqueeze(1)
