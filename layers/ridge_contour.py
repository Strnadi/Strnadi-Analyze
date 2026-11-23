"""Sub-bin ridge (dominant tone) track per frame.

For each frame: the strongest bin, refined by parabolic interpolation of its two neighbours (sub-bin
frequency, so 100-250 Hz criteria are resolved even on a 60 Hz grid), its level, its prominence over
the frame median (tonalness), the frame-to-frame frequency change, plus coarse band energies.
(B, C, F, T) -> (B, 4 + n_bands, T).
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
        """Spectrogram (B, C, F, T) -> per-frame features (B, 4 + n_bands, T); uses channel 0."""
        level = spec[:, 0]  # (B, F, T)
        n_bins = level.shape[1]

        # the strongest bin of every frame and the levels of its two neighbours
        peak_level, peak_bin = level.max(dim=1, keepdim=True)  # both (B, 1, T)
        bin_below = (peak_bin - 1).clamp_min(0)
        bin_above = (peak_bin + 1).clamp_max(n_bins - 1)
        level_below = level.gather(1, bin_below)  # (B, 1, T)
        level_above = level.gather(1, bin_above)  # (B, 1, T)

        # parabola through the three points: its vertex lies `offset` bins away from the peak bin
        curvature = level_below - 2 * peak_level + level_above
        offset = torch.where(
            curvature.abs() > 1e-6,
            0.5 * (level_below - level_above) / curvature,
            torch.zeros_like(curvature),
        )
        offset = offset.clamp(-0.5, 0.5)
        # no refinement when the peak is the first or the last bin (one neighbour is missing)
        peak_at_edge = (peak_bin == 0) | (peak_bin == n_bins - 1)
        offset = torch.where(peak_at_edge, torch.zeros_like(offset), offset)

        # sub-bin position of the ridge, scaled from bins to [-1, 1]
        ridge_freq = (peak_bin.float() + offset) / (n_bins - 1) * 2 - 1  # (B, 1, T)
        frame_median = level.median(dim=1, keepdim=True).values  # (B, 1, T)
        prominence = peak_level - frame_median  # tonalness
        # frame-to-frame change of the ridge frequency; the first frame gets 0
        freq_change = F.pad(ridge_freq[..., 1:] - ridge_freq[..., :-1], (1, 0))  # (B, 1, T)

        # coarse band energies: the F bins averaged down to n_bands
        level_time_first = level.transpose(1, 2)  # (B, T, F)
        band_energies = F.adaptive_avg_pool1d(level_time_first, self.n_bands)  # (B, T, n_bands)
        band_energies = band_energies.transpose(1, 2)  # (B, n_bands, T)

        return torch.cat([ridge_freq, peak_level, prominence, freq_change, band_energies], 1)
