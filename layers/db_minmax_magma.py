"""dB spectrogram -> per-example min-max -> 'magma' colour map (3 channels), for ImageNet-pretrained backbones.

Origin: legacy `layers/mel_to_magma.py` (production DenseNet front-end). Kept for the legacy control models only:
the colour map adds no information over one channel, and min-max scaling makes the whole input depend on the
single loudest and quietest pixel.
"""
import torch
import torch.nn as nn


class DbMinMaxMagma(nn.Module):
    out_channels = 3

    def __init__(self, n_colors=256, top_db=80.0):
        super().__init__()
        import matplotlib
        lut = matplotlib.colormaps['magma'].resampled(n_colors)(range(n_colors))[:, :3]
        self.register_buffer('lut', torch.tensor(lut, dtype=torch.float32), persistent=False)
        self.top_db = top_db

    def forward(self, bands):  # (B, F, T) power -> (B, 3, F, T)
        db = 10.0 * torch.log10(bands.clamp_min(1e-10))
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - self.top_db)
        lo, hi = db.amin(dim=(1, 2), keepdim=True), db.amax(dim=(1, 2), keepdim=True)
        idx = ((db - lo) / (hi - lo + 1e-6) * (self.lut.shape[0] - 1)).round().long()
        return self.lut[idx].permute(0, 3, 1, 2)
