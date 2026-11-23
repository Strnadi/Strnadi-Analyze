"""waveform -> spectrogram front-end = StftPower -> filterbank -> normalisation.

A pure composition container: pass any filterbank layer (mel / linear) and any normalisation layer
(log-median, log-percentile, PCEN, ...). Output (B, C, F, T) with T = N / hop.
"""
import torch.nn as nn
from .stft_power import StftPower


class SpecFrontend(nn.Module):
    def __init__(self, bank, norm, n_fft=1024, hop=320):
        super().__init__()
        self.stft = StftPower(n_fft, hop, *bank.bin_range)
        self.bank, self.norm = bank, norm
        self.hop, self.n_bins, self.out_channels = hop, bank.n_bins, norm.out_channels

    def forward(self, wave):
        return self.norm(self.bank(self.stft(wave)))
