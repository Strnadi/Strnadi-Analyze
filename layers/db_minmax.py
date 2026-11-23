"""dB spectrogram min-max scaled to [0, 1] per example (1 channel): the input/target scaling the
legacy autoencoders assume (sigmoid output + MSE).
Origin: legacy `processing/encode.py` + `model/autoencoder.py`."""

import torch
import torch.nn as nn


class DbMinMax(nn.Module):
    out_channels = 1

    def __init__(self, top_db=80.0):
        super().__init__()
        self.top_db = top_db

    def forward(self, bands):
        """Band power (B, F, T) -> (B, 1, F, T) in [0, 1]."""
        db = 10.0 * torch.log10(bands.clamp_min(1e-10))  # (B, F, T)
        # limit the dynamic range to top_db below the loudest cell of each example
        loudest = db.amax(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        db = torch.maximum(db, loudest - self.top_db)

        # min-max scale each example to [0, 1]
        lowest = db.amin(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        highest = db.amax(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        scaled = (db - lowest) / (highest - lowest + 1e-6)
        return scaled.unsqueeze(1)  # (B, 1, F, T)
