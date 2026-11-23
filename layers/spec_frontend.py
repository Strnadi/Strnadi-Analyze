"""waveform -> spectrogram front-end = StftPower -> filterbank -> normalisation.

A pure composition container: pass any filterbank layer (mel / linear) and any normalisation layer
(log-median, log-percentile, PCEN, ...). Output (B, C, F, T) with T = N / hop.
"""

import torch.nn as nn

from .stft_power import StftPower


class SpecFrontend(nn.Module):
    def __init__(self, bank, norm, n_fft=1024, hop=320):
        super().__init__()
        # compute only the STFT bins the filterbank uses
        first_bin, end_bin = bank.bin_range
        self.stft = StftPower(n_fft, hop, first_bin, end_bin)
        self.bank = bank
        self.norm = norm
        self.hop = hop
        self.n_bins = bank.n_bins
        self.out_channels = norm.out_channels

    def forward(self, wave):
        """Waveform (B, N) -> spectrogram (B, C, F, T) with T = N / hop."""
        power = self.stft(wave)  # (B, STFT bins in bin_range, T)
        bands = self.bank(power)  # (B, F, T)
        return self.norm(bands)  # (B, C, F, T)
