"""Triangular filters equally spaced in Hz.

Dialect criteria are absolute Hz offsets (100-250 Hz bandwidth, 1000-2000 Hz steps; Diblikova et
al. 2023 measured at 86 Hz resolution), so a linear scale keeps them uniform over the band, and a
shift along this axis is an exact Hz translation (a label-safe pitch augmentation).

Input: power STFT restricted to `bin_range`; output (B, n_bins, T).
Origin: research session (written, never run) + its literature report + both reviews.
"""

import numpy as np
import torch
import torch.nn as nn


class LinearFilterbank(nn.Module):
    def __init__(self, n_bins=192, fmin=2000.0, fmax=9500.0, n_fft=1024, sr=32000):
        super().__init__()
        self.fmin, self.fmax = fmin, fmax  # band edges in Hz (used to label plots)
        fft_freqs = np.fft.rfftfreq(n_fft, 1 / sr)  # (n_fft // 2 + 1,) centre of every STFT bin, Hz
        # n_bins + 2 equally spaced points: filter i rises from point i to its peak at point i + 1
        # and falls back to zero at point i + 2
        edges = np.linspace(fmin, fmax, n_bins + 2)
        weights = np.zeros((n_bins, len(fft_freqs)), np.float32)
        for i in range(n_bins):
            left = edges[i]
            centre = edges[i + 1]
            right = edges[i + 2]
            rising = (fft_freqs - left) / (centre - left)
            falling = (right - fft_freqs) / (right - centre)
            weights[i] = np.clip(np.minimum(rising, falling), 0, None)

        has_empty_filter = (weights.sum(1) == 0).any()  # a filter narrower than one STFT bin
        if has_empty_filter:
            raise ValueError(
                f'{n_bins} linear bins over {fmin}-{fmax} Hz is finer than the STFT '
                f'(n_fft={n_fft}); raise n_fft'
            )
        weights /= weights.sum(1, keepdims=True)  # every filter sums to 1

        # only the STFT bins that some filter uses are needed: [first, last + 1)
        used_fft_bins = np.nonzero(weights.sum(0) > 0)[0]
        first_bin = int(used_fft_bins.min())
        end_bin = int(used_fft_bins.max()) + 1
        self.bin_range = (first_bin, end_bin)
        self.n_bins = n_bins
        used_weights = weights[:, first_bin:end_bin].copy()  # (n_bins, end_bin - first_bin)
        self.register_buffer('fb', torch.from_numpy(used_weights), persistent=False)

    def forward(self, power):
        """Power STFT (B, bins, T) restricted to `bin_range` -> band power (B, n_bins, T)."""
        return torch.matmul(self.fb, power)
