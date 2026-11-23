"""MobileNetV2-style bottleneck autoencoder, ported from Keras: stride-2 stem -> MBConv stages (downsample by 8)
-> spatial attention -> global average pool -> `latent_dim` vector -> transposed-conv decoder (sigmoid, MSE).
Returns dict(emb, recon, target).

Unlike the Keras script, the decoder volume is derived from the input size (the script hard-coded 16 x 47,
which only fits a 128 x 376 input). `n_bins` and `n_frames` must be divisible by 8.
Origin: legacy `model/autoencoder.py::build_mobile_autoencoder`.
"""
import torch.nn as nn
from layers.mbconv import MBConv
from layers.spatial_attention import SpatialAttention
from .conv_autoencoder import minmax_mel_frontend


class MobileAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, n_bins=128, n_frames=400, n_classes=0):
        super().__init__()
        assert n_bins % 8 == 0 and n_frames % 8 == 0
        self.frontend = minmax_mel_frontend(n_bins)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            MBConv(32, 16, expand=1), MBConv(16, 24, stride=2), MBConv(24, 24), MBConv(24, 32, stride=2), MBConv(32, 32),
            MBConv(32, 64), SpatialAttention(7))
        self.to_latent = nn.Linear(64, latent_dim)
        self.vol = (64, n_bins // 8, n_frames // 8)
        self.from_latent = nn.Linear(latent_dim, self.vol[0] * self.vol[1] * self.vol[2])
        def up(ci, co): return [nn.ConvTranspose2d(ci, co, 3, 2, 1, output_padding=1), nn.ReLU(inplace=True)]
        self.decoder = nn.Sequential(*up(64, 32), *up(32, 24), *up(24, 16), nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        z = self.to_latent(self.encoder(spec).mean(dim=(2, 3)))
        recon = self.decoder(self.from_latent(z).view(-1, *self.vol))
        return {'emb': z, 'recon': recon, 'target': spec, 'logits': None}
