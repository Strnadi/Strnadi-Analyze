"""MobileNetV2-style bottleneck autoencoder, ported from Keras: stride-2 stem -> MBConv stages
(downsample by 8) -> spatial attention -> global average pool -> `latent_dim` vector ->
transposed-conv decoder (sigmoid, MSE). Returns dict(emb, recon, target).

Unlike the Keras script, the decoder volume is derived from the input size (the script hard-coded
16 x 47, which only fits a 128 x 376 input). `n_bins` and `n_frames` must be divisible by 8.

Origin: legacy `model/autoencoder.py::build_mobile_autoencoder`.
"""

import torch.nn as nn

from layers.mbconv import MBConv
from layers.spatial_attention import SpatialAttention
from model.base import Model

from .conv_autoencoder import minmax_mel_frontend


class MobileAutoencoder(Model):
    """Spectrogram -> small latent vector -> reconstructed spectrogram, with a MobileNetV2 encoder.

    Args:
        latent_dim: size of the bottleneck vector (the embedding).
        n_bins: number of mel bands of the input; must be divisible by 8.
        n_frames: number of spectrogram frames the decoder produces (400 = 4 s); must be divisible
            by 8. Feed clips of exactly this length, so that `recon` and `target` match.
        n_classes: not used (the autoencoder has no classifier); accepted and ignored.

    `forward` returns a dict: `emb` (B, latent_dim), `recon` and `target` (both (B, 1, n_bins,
    n_frames), values in [0, 1]; train with MSE between them) and `logits` = None.
    """

    def __init__(self, latent_dim=64, n_bins=128, n_frames=400, n_classes=0):
        super().__init__()
        assert n_bins % 8 == 0 and n_frames % 8 == 0

        # --- front-end: waveform -> mel spectrogram in [0, 1] (also the reconstruction target)
        self.frontend = minmax_mel_frontend(n_bins)

        # --- encoder: stride-2 stem, then MBConv blocks; frequency and time shrink by 8 in total
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            MBConv(32, 16, expand=1),
            MBConv(16, 24, stride=2),
            MBConv(24, 24),
            MBConv(24, 32, stride=2),
            MBConv(32, 32),
            MBConv(32, 64),
            SpatialAttention(7),
        )

        # --- bottleneck: the globally averaged feature map (64 values) -> `latent_dim`, then back
        #     to a volume the decoder can upsample
        self.to_latent = nn.Linear(64, latent_dim)
        self.vol = (64, n_bins // 8, n_frames // 8)  # (channels, freq, time) entering the decoder
        volume_size = self.vol[0] * self.vol[1] * self.vol[2]
        self.from_latent = nn.Linear(latent_dim, volume_size)

        # --- decoder: three transposed convolutions, each doubling frequency and time; sigmoid out
        def upsample_x2(in_channels, out_channels):
            return [
                nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, output_padding=1),
                nn.ReLU(inplace=True),
            ]

        self.decoder = nn.Sequential(
            *upsample_x2(64, 32),
            *upsample_x2(32, 24),
            *upsample_x2(24, 16),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 128, 400)   values in [0, 1]
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`

        # --- encode
        feature_map = self.encoder(spectrogram)  # (B, 64, 16, 50)
        averaged = feature_map.mean(dim=(2, 3))  # (B, 64)   global average pool
        latent = self.to_latent(averaged)  # (B, 64)

        # --- decode
        flat = self.from_latent(latent)  # (B, 51200)
        decoder_volume = flat.view(-1, *self.vol)  # (B, 64, 16, 50)
        reconstruction = self.decoder(decoder_volume)  # (B, 1, 128, 400)   values in [0, 1]

        return {'emb': latent, 'recon': reconstruction, 'target': spectrogram, 'logits': None}


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(MobileAutoencoder())
