"""Bidirectional GRU over the frame series, (B, D, T) -> (B, D, T).
Origin: legacy `model/densenet_simclr.py` (CRNN) and research `temporal='gru'`."""

import torch.nn as nn


class TemporalBiGRU(nn.Module):
    def __init__(self, d, n_layers=1, d_in=None):
        super().__init__()
        # each direction has d // 2 units, so the two concatenated directions give d (for even d)
        self.gru = nn.GRU(
            input_size=d_in or d,
            hidden_size=d // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        """Frame series (B, D_in, T) -> (B, D, T)."""
        time_first = x.transpose(1, 2)  # (B, T, D_in)
        outputs, _last_hidden = self.gru(time_first)  # outputs: (B, T, D)
        return outputs.transpose(1, 2)  # (B, D, T)
