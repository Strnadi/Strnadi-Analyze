"""Per-channel energy normalisation (Wang et al. 2017; Lostanlen et al. 2019 for bioacoustics).

Adaptive gain control + root compression: robust to loudness, distance and stationary noise, i.e.
the far-field / faint-terminal-element regime. The IIR smoother is realised as a truncated FIR
(exports as a conv).
Origin: research session code (never run there); recommended by its own literature report and both
reviews.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCEN(nn.Module):
    out_channels = 1

    def __init__(self, s=0.04, alpha=0.98, delta=2.0, r=0.5, taps=200, eps=1e-6, trainable=False):
        super().__init__()
        self.s = s  # smoothing coefficient of the IIR smoother
        self.taps = taps  # length of its truncated FIR form, in frames
        self.eps = eps
        # alpha (gain exponent), delta (bias) and r (root), stored as logarithms
        self.log_adr = nn.Parameter(torch.tensor([alpha, delta, r]).log(), requires_grad=trainable)

    def forward(self, bands):
        """Band power (B, F, T) -> (B, 1, F, T)."""
        B, n_bins, T = bands.shape
        alpha, delta, r = self.log_adr.exp()

        # FIR form of the smoother: the frame `lag` steps in the past has weight s * (1 - s) ** lag
        lags = torch.arange(self.taps, device=bands.device, dtype=bands.dtype)  # (taps,)
        impulse_response = self.s * (1 - self.s) ** lags  # (taps,), most recent frame first
        # conv1d slides the kernel forward in time, so the oldest frame has to come first
        smoother_kernel = impulse_response.flip(0)[None, None]  # (1, 1, taps)

        # smooth every band of every example as its own 1-channel sequence
        band_sequences = bands.reshape(B * n_bins, 1, T)  # (B * F, 1, T)
        # repeat the first frame taps - 1 times on the left so every frame has a full history
        padded = F.pad(band_sequences, (self.taps - 1, 0), mode='replicate')
        smoothed = F.conv1d(padded, smoother_kernel).reshape(B, n_bins, T)  # (B, F, T)

        gain_controlled = bands / (self.eps + smoothed) ** alpha
        compressed = (gain_controlled + delta) ** r - delta**r
        return compressed.unsqueeze(1)  # (B, 1, F, T)
