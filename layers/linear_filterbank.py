"""Triangular filters equally spaced in Hz.

Dialect criteria are absolute Hz offsets (100-250 Hz bandwidth, 1000-2000 Hz steps; Diblikova et al. 2023 measured
at 86 Hz resolution), so a linear scale keeps them uniform over the band, and a shift along this axis is an exact
Hz translation (a label-safe pitch augmentation).
Origin: research session (written, never run) + its literature report + both reviews.
"""
import numpy as np
import torch
import torch.nn as nn


class LinearFilterbank(nn.Module):
    def __init__(self, n_bins=192, fmin=2000.0, fmax=9500.0, n_fft=1024, sr=32000):
        super().__init__()
        freqs = np.fft.rfftfreq(n_fft, 1 / sr); c = np.linspace(fmin, fmax, n_bins + 2)
        w = np.zeros((n_bins, len(freqs)), np.float32)
        for i in range(n_bins):
            w[i] = np.clip(np.minimum((freqs - c[i]) / (c[i + 1] - c[i]), (c[i + 2] - freqs) / (c[i + 2] - c[i + 1])), 0, None)
        if (w.sum(1) == 0).any():
            raise ValueError(f'{n_bins} linear bins over {fmin}-{fmax} Hz is finer than the STFT (n_fft={n_fft}); raise n_fft')
        w /= w.sum(1, keepdims=True)
        used = np.nonzero(w.sum(0) > 0)[0]
        self.bin_range = (int(used.min()), int(used.max()) + 1)
        self.n_bins = n_bins
        self.register_buffer('fb', torch.from_numpy(w[:, self.bin_range[0]:self.bin_range[1]].copy()), persistent=False)

    def forward(self, power):
        return torch.matmul(self.fb, power)
