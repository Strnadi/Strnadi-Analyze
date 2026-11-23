"""Convolutional bottleneck autoencoder with asymmetric pooling and spatial attention, ported from Keras.

128-bin mel in [0, 1] -> conv32 / pool(2,2) -> conv64 / pool(time only) -> conv128 -> spatial attention ->
pool(2,2) -> flatten -> `latent_dim` vector -> mirrored decoder with a sigmoid output; MSE reconstruction.
Returns dict(emb, recon, target). The latent is meant for unsupervised clustering (HDBSCAN / GMM).

Caveat recorded by the research session: plain reconstruction spends its capacity on background and loudness,
while a dialect is a few hundred Hz in one or two elements. Unlike the Keras script (fixed 128 x 376 input),
shapes are derived from `n_frames`, which must be divisible by 8.
Origin: legacy `model/autoencoder.py::build_yellowhammer_autoencoder`.
"""
import torch
import torch.nn as nn
from layers.db_minmax import DbMinMax
from layers.spec_frontend import SpecFrontend
from layers.mel_filterbank import MelFilterbank
from layers.spatial_attention import SpatialAttention


def minmax_mel_frontend(n_bins=128, fmin=1500.0, fmax=11000.0, n_fft=1024, hop=320):
    return SpecFrontend(MelFilterbank(n_bins, fmin, fmax, n_fft), DbMinMax(), n_fft, hop)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64, n_bins=128, n_frames=400, n_classes=0):
        super().__init__()
        assert n_bins % 4 == 0 and n_frames % 8 == 0
        self.frontend = minmax_mel_frontend(n_bins)
        def block(ci, co): return [nn.Conv2d(ci, co, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(co)]
        self.encoder = nn.Sequential(*block(1, 32), nn.MaxPool2d((2, 2)), *block(32, 64), nn.MaxPool2d((1, 2)),
                                     *block(64, 128), SpatialAttention(7), nn.MaxPool2d((2, 2)))
        self.vol = (128, n_bins // 4, n_frames // 8)
        n = self.vol[0] * self.vol[1] * self.vol[2]
        self.to_latent, self.from_latent = nn.Linear(n, latent_dim), nn.Linear(latent_dim, n)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Upsample(scale_factor=(2, 2)),
            nn.ConvTranspose2d(128, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Upsample(scale_factor=(1, 2)),
            nn.ConvTranspose2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(32, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        z = self.to_latent(self.encoder(spec).flatten(1))
        recon = self.decoder(self.from_latent(z).view(-1, *self.vol))
        return {'emb': z, 'recon': recon, 'target': spec, 'logits': None}
