"""Several front-ends with different STFT window lengths, stacked as input channels.

Why: the dialect-defining terminal elements are long, thin, faint tones; the first phrase is fast syllables.
A long window (64-128 ms) integrates a steady tone and gains several dB of SNR on exactly the element the
classifier misses (complete-vs-Unfinished AUC for BhBl is only 0.79-0.83), while a short window keeps syllable
timing. All members must share hop and bin count. Origin: fork review (architecture idea 1).
"""
import torch
import torch.nn as nn


class MultiResFrontend(nn.Module):
    def __init__(self, frontends):
        super().__init__()
        self.members = nn.ModuleList(frontends)
        assert len({f.hop for f in frontends}) == 1 and len({f.n_bins for f in frontends}) == 1
        self.hop, self.n_bins = frontends[0].hop, frontends[0].n_bins
        self.out_channels = sum(f.out_channels for f in frontends)

    def forward(self, wave):
        return torch.cat([f(wave) for f in self.members], 1)
