"""Band-limited mel filterbank (HTK mel scale, Slaney area normalisation; identical to
`librosa.filters.mel(htk=True, norm='slaney')`, without the librosa dependency).

Input: power STFT restricted to `bin_range`; output (B, n_bins, T).
Origin: research session default front-end (128 bins, 1.5-11 kHz).
"""
import numpy as np
import torch
import torch.nn as nn


def mel_weights(sr, n_fft, n_bins, fmin, fmax):
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f, np.float64) / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (np.asarray(m, np.float64) / 2595.0) - 1.0)
    fft_f = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_bins + 2))
    fdiff = np.diff(mel_f); ramps = mel_f[:, None] - fft_f[None]
    w = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    w *= (2.0 / (mel_f[2:] - mel_f[:-2]))[:, None]
    return w.astype(np.float32)


class MelFilterbank(nn.Module):
    def __init__(self, n_bins=128, fmin=1500.0, fmax=11000.0, n_fft=1024, sr=32000):
        super().__init__()
        w = mel_weights(sr, n_fft, n_bins, fmin, fmax)
        used = np.nonzero(w.sum(0) > 0)[0]
        self.bin_range = (int(used.min()), int(used.max()) + 1)
        self.n_bins = n_bins
        self.register_buffer('fb', torch.from_numpy(w[:, self.bin_range[0]:self.bin_range[1]].copy()), persistent=False)

    def forward(self, power):  # (B, bins, T) -> (B, n_bins, T)
        return torch.matmul(self.fb, power)
