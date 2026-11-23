"""Several front-ends with different STFT window lengths, stacked as input channels.

Why: the dialect-defining terminal elements are long, thin, faint tones; the first phrase is fast
syllables. A long window (64-128 ms) integrates a steady tone and gains several dB of SNR on exactly
the element the classifier misses (complete-vs-Unfinished AUC for BhBl is only 0.79-0.83), while a
short window keeps syllable timing. All members must share hop and bin count.
Origin: fork review (architecture idea 1).
"""

import torch
import torch.nn as nn


class MultiResFrontend(nn.Module):
    def __init__(self, frontends):
        super().__init__()
        self.members = nn.ModuleList(frontends)
        # the members' outputs are concatenated along channels, so F and T must agree
        hops = {frontend.hop for frontend in frontends}
        bin_counts = {frontend.n_bins for frontend in frontends}
        assert len(hops) == 1 and len(bin_counts) == 1
        self.hop = frontends[0].hop
        self.n_bins = frontends[0].n_bins
        self.out_channels = sum(frontend.out_channels for frontend in frontends)

    def forward(self, wave):
        """Waveform (B, N) -> (B, sum of the members' channels, F, T)."""
        member_specs = [frontend(wave) for frontend in self.members]  # each (B, C_i, F, T)
        return torch.cat(member_specs, 1)
