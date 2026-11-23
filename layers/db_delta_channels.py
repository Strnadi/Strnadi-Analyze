"""dB spectrogram + its first and second time derivatives as 3 channels, each min-max scaled to [0, 1].

Origin: legacy `model/mobilenet.py` ("physics channels": mel, delta, delta-delta via the fixed kernels
[-0.5, 0, 0.5] and [1, -2, 1] along time, then `rescale`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DbDeltaChannels(nn.Module):
    out_channels = 3

    def __init__(self, top_db=80.0):
        super().__init__()
        self.top_db = top_db
        self.register_buffer('k', torch.tensor([[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]]).view(2, 1, 1, 3), persistent=False)

    def forward(self, bands):  # (B, F, T) power -> (B, 3, F, T)
        db = 10.0 * torch.log10(bands.clamp_min(1e-10))
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - self.top_db).unsqueeze(1)
        d = F.conv2d(F.pad(db, (1, 1, 0, 0), mode='replicate'), self.k)
        x = torch.cat([db, d], 1)
        lo, hi = x.amin(dim=(2, 3), keepdim=True), x.amax(dim=(2, 3), keepdim=True)
        return (x - lo) / (hi - lo + 1e-5)
