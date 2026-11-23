"""Convolutional bottleneck autoencoder with asymmetric pooling and spatial attention, ported from
Keras.

128-bin mel in [0, 1] -> conv32 / pool(2,2) -> conv64 / pool(time only) -> conv128 -> spatial
attention -> pool(2,2) -> flatten -> `latent_dim` vector -> mirrored decoder with a sigmoid output;
MSE reconstruction. Returns dict(emb, recon, target). The latent is meant for unsupervised
clustering (HDBSCAN / GMM).

Caveat recorded by the research session: plain reconstruction spends its capacity on background and
loudness, while a dialect is a few hundred Hz in one or two elements. Unlike the Keras script
(fixed 128 x 376 input), shapes are derived from `n_frames`, which must be divisible by 8.

Origin: legacy `model/autoencoder.py::build_yellowhammer_autoencoder`.
"""

import torch.nn as nn

from layers.db_minmax import DbMinMax
from layers.mel_filterbank import MelFilterbank
from layers.spatial_attention import SpatialAttention
from layers.spec_frontend import SpecFrontend
from model.base import Model


def minmax_mel_frontend(n_bins=128, fmin=1500.0, fmax=11000.0, n_fft=1024, hop=320):
    """Front-end of the autoencoders: waveform -> mel spectrogram in dB, scaled to [0, 1].

    Args:
        n_bins: number of mel bands.
        fmin: lowest frequency of the mel bank in Hz.
        fmax: highest frequency of the mel bank in Hz.
        n_fft: STFT window length in samples.
        hop: STFT hop in samples (320 = 10 ms at 32 kHz).
    """
    filterbank = MelFilterbank(n_bins, fmin, fmax, n_fft)
    normalisation = DbMinMax()
    return SpecFrontend(filterbank, normalisation, n_fft, hop)


class ConvAutoencoder(Model):
    """Spectrogram -> small latent vector -> reconstructed spectrogram.

    Args:
        latent_dim: size of the bottleneck vector (the embedding).
        n_bins: number of mel bands of the input; must be divisible by 4.
        n_frames: number of spectrogram frames of the input (400 = 4 s); must be divisible by 8.
            The model only accepts clips of exactly this length.
        n_classes: not used (the autoencoder has no classifier); accepted and ignored.

    `forward` returns a dict: `emb` (B, latent_dim), `recon` and `target` (both (B, 1, n_bins,
    n_frames), values in [0, 1]; train with MSE between them) and `logits` = None.
    """

    def __init__(self, latent_dim=64, n_bins=128, n_frames=400, n_classes=0):
        super().__init__()
        assert n_bins % 4 == 0 and n_frames % 8 == 0

        # --- front-end: waveform -> mel spectrogram in [0, 1] (also the reconstruction target)
        self.frontend = minmax_mel_frontend(n_bins)

        # --- encoder: three conv stages; the middle pooling halves time only
        def conv_relu_bn(in_channels, out_channels):
            return [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            ]

        self.encoder = nn.Sequential(
            *conv_relu_bn(1, 32),
            nn.MaxPool2d((2, 2)),
            *conv_relu_bn(32, 64),
            nn.MaxPool2d((1, 2)),
            *conv_relu_bn(64, 128),
            SpatialAttention(7),
            nn.MaxPool2d((2, 2)),
        )

        # --- bottleneck: flatten the encoder volume, squeeze it to `latent_dim` and back
        # the encoder pools frequency by 4 and time by 8
        self.vol = (128, n_bins // 4, n_frames // 8)  # (channels, freq, time) entering the decoder
        volume_size = self.vol[0] * self.vol[1] * self.vol[2]
        self.to_latent = nn.Linear(volume_size, latent_dim)
        self.from_latent = nn.Linear(latent_dim, volume_size)

        # --- decoder: mirror of the encoder, ending in a sigmoid (output in [0, 1])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2, 2)),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(1, 2)),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 128, 400)   values in [0, 1]
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`

        # --- encode
        encoded_volume = self.encoder(spectrogram)  # (B, 128, 32, 50)
        flat = encoded_volume.flatten(1)  # (B, 204800)
        latent = self.to_latent(flat)  # (B, 64)

        # --- decode
        flat = self.from_latent(latent)  # (B, 204800)
        decoder_volume = flat.view(-1, *self.vol)  # (B, 128, 32, 50)
        reconstruction = self.decoder(decoder_volume)  # (B, 1, 128, 400)   values in [0, 1]

        return {'emb': latent, 'recon': reconstruction, 'target': spectrogram, 'logits': None}


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(ConvAutoencoder())
