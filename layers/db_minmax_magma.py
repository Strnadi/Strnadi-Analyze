"""dB spectrogram -> per-example min-max -> 'magma' colour map (3 channels), for
ImageNet-pretrained backbones.

Origin: legacy `layers/mel_to_magma.py` (production DenseNet front-end). Kept for the legacy
control models only: the colour map adds no information over one channel, and min-max scaling
makes the whole input depend on the single loudest and quietest pixel.
"""

import torch
import torch.nn as nn


class DbMinMaxMagma(nn.Module):
    out_channels = 3

    def __init__(self, n_colors=256, top_db=80.0):
        super().__init__()
        import matplotlib

        colormap = matplotlib.colormaps['magma'].resampled(n_colors)
        rgba = colormap(range(n_colors))  # (n_colors, 4) array
        rgb = rgba[:, :3]  # drop the alpha column
        # lut = colour look-up table, (n_colors, 3)
        self.register_buffer('lut', torch.tensor(rgb, dtype=torch.float32), persistent=False)
        self.top_db = top_db

    def forward(self, bands):
        """Band power (B, F, T) -> colour image (B, 3, F, T)."""
        db = 10.0 * torch.log10(bands.clamp_min(1e-10))  # (B, F, T)
        # limit the dynamic range to top_db below the loudest cell of each example
        loudest = db.amax(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        db = torch.maximum(db, loudest - self.top_db)

        # min-max scale each example to [0, 1]
        lowest = db.amin(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        highest = db.amax(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        scaled = (db - lowest) / (highest - lowest + 1e-6)

        # look every cell up in the colour table
        n_colors = self.lut.shape[0]
        color_index = (scaled * (n_colors - 1)).round().long()  # (B, F, T) integers
        rgb = self.lut[color_index]  # (B, F, T, 3)
        return rgb.permute(0, 3, 1, 2)  # (B, 3, F, T)
