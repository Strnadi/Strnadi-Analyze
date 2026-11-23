"""Small pre-norm transformer encoder over the frame series with a learned position embedding,
(B, D, T) -> (B, D, T). Origin: research `temporal='attn'`."""

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    def __init__(self, d, n_layers=3, heads=4, drop=0.1, max_len=512):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=heads,
            dim_feedforward=2 * d,
            dropout=drop,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d))  # learned position embedding

    def forward(self, x):
        """Frame series (B, D, T) -> (B, D, T)."""
        n_frames = x.shape[-1]
        tokens = x.transpose(1, 2)  # (B, T, D)
        tokens = tokens + self.pos[:, :n_frames]  # add the position of every frame
        encoded = self.enc(tokens)  # (B, T, D)
        return encoded.transpose(1, 2)  # (B, D, T)
