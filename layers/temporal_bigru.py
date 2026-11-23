"""Bidirectional GRU over the frame series, (B, D, T) -> (B, D, T).
Origin: legacy `model/densenet_simclr.py` (CRNN) and research `temporal='gru'`."""
import torch.nn as nn


class TemporalBiGRU(nn.Module):
    def __init__(self, d, n_layers=1, d_in=None):
        super().__init__()
        self.gru = nn.GRU(d_in or d, d // 2, num_layers=n_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x.transpose(1, 2))[0].transpose(1, 2)
