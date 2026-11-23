"""dB spectrogram min-max scaled to [0, 1] per example (1 channel): the input/target scaling the legacy
autoencoders assume (sigmoid output + MSE). Origin: legacy `processing/encode.py` + `model/autoencoder.py`."""
import torch
import torch.nn as nn


class DbMinMax(nn.Module):
    out_channels = 1

    def __init__(self, top_db=80.0):
        super().__init__(); self.top_db = top_db

    def forward(self, bands):  # (B, F, T) power -> (B, 1, F, T) in [0, 1]
        db = 10.0 * torch.log10(bands.clamp_min(1e-10))
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - self.top_db)
        lo, hi = db.amin(dim=(1, 2), keepdim=True), db.amax(dim=(1, 2), keepdim=True)
        return ((db - lo) / (hi - lo + 1e-6)).unsqueeze(1)
