"""Per-channel energy normalisation (Wang et al. 2017; Lostanlen et al. 2019 for bioacoustics).

Adaptive gain control + root compression: robust to loudness, distance and stationary noise, i.e. the
far-field / faint-terminal-element regime. The IIR smoother is realised as a truncated FIR (exports as a conv).
Origin: research session code (never run there); recommended by its own literature report and both reviews.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PCEN(nn.Module):
    out_channels = 1

    def __init__(self, s=0.04, alpha=0.98, delta=2.0, r=0.5, taps=200, eps=1e-6, trainable=False):
        super().__init__()
        self.s, self.taps, self.eps = s, taps, eps
        self.log_adr = nn.Parameter(torch.tensor([alpha, delta, r]).log(), requires_grad=trainable)

    def forward(self, bands):  # (B, F, T) power -> (B, 1, F, T)
        B, Fq, T = bands.shape
        alpha, delta, r = self.log_adr.exp()
        k = (self.s * (1 - self.s) ** torch.arange(self.taps, device=bands.device, dtype=bands.dtype)).flip(0)[None, None]
        m = F.conv1d(F.pad(bands.reshape(B * Fq, 1, T), (self.taps - 1, 0), mode='replicate'), k).reshape(B, Fq, T)
        y = (bands / (self.eps + m) ** alpha + delta) ** r - delta ** r
        return y.unsqueeze(1)
