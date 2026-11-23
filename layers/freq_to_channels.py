"""Fold the remaining frequency bins into channels: (B, C, F, T) -> (B, C * F, T).

Unlike frequency pooling this keeps absolute pitch ("Bh" vs "Bl" is higher vs lower), and hands a
1-D frame series to the temporal module.
Origin: research session; the legacy SimCLR CRNN does the same before its GRU.
"""

import torch.nn as nn


class FreqToChannels(nn.Module):
    def forward(self, x):
        """(B, C, F, T) -> (B, C * F, T)."""
        B, C, n_bins, T = x.shape
        return x.reshape(B, C * n_bins, T)
