"""Sub-bin ridge (dominant tone) track per frame.

For each frame: the strongest bin, refined by parabolic interpolation of its two neighbours (sub-bin frequency,
so 100-250 Hz criteria are resolved even on a 60 Hz grid), its level, its prominence over the frame median
(tonalness), the frame-to-frame frequency change, plus coarse band energies. (B, C, F, T) -> (B, 4 + n_bands, T).
Use on a LINEAR-frequency spectrogram so that the frequency axis is in Hz.
Origin: fork review (architecture idea 5), replacing the bin-quantised `ContourTopK`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RidgeContour(nn.Module):
    def __init__(self, n_bands=8):
        super().__init__()
        self.n_bands = n_bands
        self.out_dim = 4 + n_bands

    def forward(self, spec):
        L = spec[:, 0]                                                # (B, F, T)
        Fq = L.shape[1]
        peak, i = L.max(dim=1, keepdim=True)                          # (B, 1, T)
        lo = L.gather(1, (i - 1).clamp_min(0)); hi = L.gather(1, (i + 1).clamp_max(Fq - 1))
        denom = (lo - 2 * peak + hi)
        delta = torch.where(denom.abs() > 1e-6, 0.5 * (lo - hi) / denom, torch.zeros_like(denom)).clamp(-0.5, 0.5)
        delta = torch.where((i == 0) | (i == Fq - 1), torch.zeros_like(delta), delta)
        freq = (i.float() + delta) / (Fq - 1) * 2 - 1
        prom = peak - L.median(dim=1, keepdim=True).values
        dfreq = F.pad(freq[..., 1:] - freq[..., :-1], (1, 0))
        bands = F.adaptive_avg_pool1d(L.transpose(1, 2), self.n_bands).transpose(1, 2)
        return torch.cat([freq, peak, prom, dfreq, bands], 1)
