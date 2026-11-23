"""Power STFT with exactly N / hop frames.

Two numerically equivalent paths: torch.stft (fast, training) and a conv1d DFT (only standard ops, so the graph
exports cleanly to ONNX). Only bins [bin_lo, bin_hi) are computed/kept.
Origin: research session (`strnadi_ml/features.py`).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StftPower(nn.Module):
    def __init__(self, n_fft=1024, hop=320, bin_lo=0, bin_hi=None):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.bin_lo, self.bin_hi = bin_lo, n_fft // 2 + 1 if bin_hi is None else bin_hi
        self.use_fft = True  # set False before export (see training/checkpoint.py::export_mode)
        self.register_buffer('win', torch.hann_window(n_fft, periodic=True), persistent=False)
        self._kernels = None

    def _dft_kernels(self):
        if self._kernels is None or self._kernels[0].device != self.win.device:
            k = torch.arange(self.bin_lo, self.bin_hi, device=self.win.device).float()[:, None]
            n = torch.arange(self.n_fft, device=self.win.device).float()[None]
            ang = 2 * math.pi * k * n / self.n_fft
            self._kernels = ((torch.cos(ang) * self.win)[:, None, :], (-torch.sin(ang) * self.win)[:, None, :])
        return self._kernels

    def forward(self, wave):  # (B, N) -> (B, bins, N // hop)
        left = (self.n_fft - self.hop) // 2
        x = F.pad(wave.unsqueeze(1), (left, self.n_fft - self.hop - left))
        if self.use_fft:
            z = torch.stft(x.squeeze(1), self.n_fft, self.hop, window=self.win, center=False, return_complex=True)
            z = z[:, self.bin_lo:self.bin_hi]
            return z.real ** 2 + z.imag ** 2
        wr, wi = self._dft_kernels()
        re, im = F.conv1d(x, wr, stride=self.hop), F.conv1d(x, wi, stride=self.hop)
        return re * re + im * im
