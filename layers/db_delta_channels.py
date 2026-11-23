"""dB spectrogram + its first and second time derivatives as 3 channels, each min-max scaled to
[0, 1].

Origin: legacy `model/mobilenet.py` ("physics channels": mel, delta, delta-delta via the fixed
kernels [-0.5, 0, 0.5] and [1, -2, 1] along time, then `rescale`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DbDeltaChannels(nn.Module):
    out_channels = 3

    def __init__(self, top_db=80.0):
        super().__init__()
        self.top_db = top_db
        delta_kernel = [-0.5, 0.0, 0.5]  # first time derivative
        delta_delta_kernel = [1.0, -2.0, 1.0]  # second time derivative
        kernels = torch.tensor([delta_kernel, delta_delta_kernel])  # (2, 3)
        # conv2d weight layout: (2 output channels, 1 input channel, 1 bin, 3 frames)
        self.register_buffer('k', kernels.view(2, 1, 1, 3), persistent=False)

    def forward(self, bands):
        """Band power (B, F, T) -> (B, 3, F, T): dB, delta, delta-delta, each in [0, 1]."""
        db = 10.0 * torch.log10(bands.clamp_min(1e-10))  # (B, F, T)
        # limit the dynamic range to top_db below the loudest cell of each example
        loudest = db.amax(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        db = torch.maximum(db, loudest - self.top_db)
        db = db.unsqueeze(1)  # (B, 1, F, T)

        # repeat the first and last frame so the 3-frame kernels keep T frames
        db_padded = F.pad(db, (1, 1, 0, 0), mode='replicate')  # (B, 1, F, T + 2)
        deltas = F.conv2d(db_padded, self.k)  # (B, 2, F, T)
        channels = torch.cat([db, deltas], 1)  # (B, 3, F, T)

        # min-max scale every channel of every example on its own
        lowest = channels.amin(dim=(2, 3), keepdim=True)  # (B, 3, 1, 1)
        highest = channels.amax(dim=(2, 3), keepdim=True)  # (B, 3, 1, 1)
        return (channels - lowest) / (highest - lowest + 1e-5)
