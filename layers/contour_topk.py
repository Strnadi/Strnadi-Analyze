"""Per-frame pitch-contour features from a normalised spectrogram: top-k peak bins (position in [-1, 1] and
level) + coarse band energies + spectral flux. (B, C, F, T) -> (B, 2k + n_bands + 1, T); uses channel 0.

The "ornithologist's view" input for a tiny 1-D model. Peak positions are quantised to one bin (60-70 Hz on the
default mel axis); `RidgeContour` is the sub-bin variant. Origin: research session (`ContourFeatures`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContourTopK(nn.Module):
    def __init__(self, k=3, n_bands=8):
        super().__init__()
        self.k, self.n_bands = k, n_bands
        self.out_dim = 2 * k + n_bands + 1

    def forward(self, spec):
        L = spec[:, 0]                                                # (B, F, T)
        Fq = L.shape[1]
        v, i = torch.topk(L, self.k, dim=1)
        pos = i.float() / (Fq - 1) * 2 - 1
        bands = F.adaptive_avg_pool1d(L.transpose(1, 2), self.n_bands).transpose(1, 2)
        flux = F.pad((L[..., 1:] - L[..., :-1]).clamp_min(0).mean(1, keepdim=True), (1, 0))
        return torch.cat([pos, v, bands, flux], 1)
