"""Power STFT with exactly N / hop frames.

Two numerically equivalent paths: torch.stft (fast, training) and a conv1d DFT (only standard ops,
so the graph exports cleanly to ONNX). Only bins [bin_lo, bin_hi) are computed/kept.
Origin: research session (`strnadi_ml/features.py`).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StftPower(nn.Module):
    def __init__(self, n_fft=1024, hop=320, bin_lo=0, bin_hi=None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop
        self.bin_lo = bin_lo
        if bin_hi is None:
            bin_hi = n_fft // 2 + 1  # keep everything up to the last (Nyquist) bin
        self.bin_hi = bin_hi
        self.use_fft = True  # set False before export (see training/checkpoint.py::export_mode)
        self.register_buffer('win', torch.hann_window(n_fft, periodic=True), persistent=False)
        self._kernels = None  # cache of `_dft_kernels`

    def _dft_kernels(self):
        """Windowed DFT basis as conv1d kernels: (real, imaginary), each (bins, 1, n_fft).

        Built on first use and again when the module has moved to another device."""
        cache_is_stale = self._kernels is None or self._kernels[0].device != self.win.device
        if cache_is_stale:
            device = self.win.device
            bin_index = torch.arange(self.bin_lo, self.bin_hi, device=device).float()[:, None]
            sample_index = torch.arange(self.n_fft, device=device).float()[None]
            # phase of bin k at sample n: 2 * pi * k * n / n_fft
            angle = 2 * math.pi * bin_index * sample_index / self.n_fft  # (bins, n_fft)
            real_kernel = (torch.cos(angle) * self.win)[:, None, :]  # (bins, 1, n_fft)
            imag_kernel = (-torch.sin(angle) * self.win)[:, None, :]  # (bins, 1, n_fft)
            self._kernels = (real_kernel, imag_kernel)
        return self._kernels

    def forward(self, wave):
        """Waveform (B, N) -> power (B, bin_hi - bin_lo, N // hop)."""
        # n_fft - hop samples of zero padding in total, split between both ends, give exactly
        # N // hop frames
        total_padding = self.n_fft - self.hop
        left = total_padding // 2
        right = total_padding - left
        padded = F.pad(wave.unsqueeze(1), (left, right))  # (B, 1, N + n_fft - hop)

        if self.use_fft:
            spectrum = torch.stft(
                padded.squeeze(1),
                self.n_fft,
                self.hop,
                window=self.win,
                center=False,
                return_complex=True,
            )  # complex (B, n_fft // 2 + 1, N // hop)
            spectrum = spectrum[:, self.bin_lo : self.bin_hi]
            return spectrum.real**2 + spectrum.imag**2

        # export path: the same DFT as two strided convolutions
        real_kernel, imag_kernel = self._dft_kernels()
        real = F.conv1d(padded, real_kernel, stride=self.hop)  # (B, bins, N // hop)
        imag = F.conv1d(padded, imag_kernel, stride=self.hop)  # (B, bins, N // hop)
        return real * real + imag * imag
