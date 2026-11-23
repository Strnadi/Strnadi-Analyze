"""Per-frame pitch-contour features from a normalised spectrogram: top-k peak bins (position in
[-1, 1] and level) + coarse band energies + spectral flux.
(B, C, F, T) -> (B, 2k + n_bands + 1, T); uses channel 0.

The "ornithologist's view" input for a tiny 1-D model. Peak positions are quantised to one bin
(60-70 Hz on the default mel axis); `RidgeContour` is the sub-bin variant.
Origin: research session (`ContourFeatures`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContourTopK(nn.Module):
    def __init__(self, k=3, n_bands=8):
        super().__init__()
        self.k = k
        self.n_bands = n_bands
        self.out_dim = 2 * k + n_bands + 1

    def forward(self, spec):
        """Spectrogram (B, C, F, T) -> per-frame features (B, 2k + n_bands + 1, T)."""
        level = spec[:, 0]  # (B, F, T), channel 0 only
        n_bins = level.shape[1]

        # the k strongest bins of every frame: their level and their position on the frequency axis
        peak_levels, peak_bins = torch.topk(level, self.k, dim=1)  # both (B, k, T)
        peak_positions = peak_bins.float() / (n_bins - 1) * 2 - 1  # bin index -> [-1, 1]

        # coarse band energies: the F bins averaged down to n_bands
        level_time_first = level.transpose(1, 2)  # (B, T, F)
        band_energies = F.adaptive_avg_pool1d(level_time_first, self.n_bands)  # (B, T, n_bands)
        band_energies = band_energies.transpose(1, 2)  # (B, n_bands, T)

        # spectral flux: the positive frame-to-frame level change, averaged over frequency
        level_rise = (level[..., 1:] - level[..., :-1]).clamp_min(0)  # (B, F, T - 1)
        flux = level_rise.mean(1, keepdim=True)  # (B, 1, T - 1)
        flux = F.pad(flux, (1, 0))  # (B, 1, T); the first frame has no predecessor and gets 0

        return torch.cat([peak_positions, peak_levels, band_energies, flux], 1)
