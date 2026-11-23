"""Band-limited mel filterbank (HTK mel scale, Slaney area normalisation; identical to
`librosa.filters.mel(htk=True, norm='slaney')`, without the librosa dependency).

Input: power STFT restricted to `bin_range`; output (B, n_bins, T).
Origin: research session default front-end (128 bins, 1.5-11 kHz).
"""

import numpy as np
import torch
import torch.nn as nn


def _hz_to_mel(hz):
    """HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, np.float64) / 700.0)


def _mel_to_hz(mel):
    """Inverse of `_hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(mel, np.float64) / 2595.0) - 1.0)


def mel_weights(sr, n_fft, n_bins, fmin, fmax):
    """Triangular mel filters as a float32 array (n_bins, n_fft // 2 + 1)."""
    fft_freqs = np.linspace(0, sr / 2, 1 + n_fft // 2)  # centre of every STFT bin, Hz
    # n_bins + 2 points equally spaced in mel: filter i rises from point i to its peak at point
    # i + 1 and falls back to zero at point i + 2
    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_bins + 2)
    edge_freqs = _mel_to_hz(mel_points)  # (n_bins + 2,) Hz
    edge_spacing = np.diff(edge_freqs)  # (n_bins + 1,) Hz between neighbouring points

    # ramps[j, f] = edge j minus STFT bin f, in Hz
    ramps = edge_freqs[:, None] - fft_freqs[None]  # (n_bins + 2, n_fft // 2 + 1)
    rising = -ramps[:-2] / edge_spacing[:-1, None]  # 0 at point i, 1 at point i + 1
    falling = ramps[2:] / edge_spacing[1:, None]  # 1 at point i + 1, 0 at point i + 2
    weights = np.maximum(0, np.minimum(rising, falling))  # (n_bins, n_fft // 2 + 1)

    # Slaney area normalisation: divide every filter by half its width in Hz
    filter_widths = edge_freqs[2:] - edge_freqs[:-2]  # (n_bins,)
    weights *= (2.0 / filter_widths)[:, None]
    return weights.astype(np.float32)


class MelFilterbank(nn.Module):
    def __init__(self, n_bins=128, fmin=1500.0, fmax=11000.0, n_fft=1024, sr=32000):
        super().__init__()
        weights = mel_weights(sr, n_fft, n_bins, fmin, fmax)
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
