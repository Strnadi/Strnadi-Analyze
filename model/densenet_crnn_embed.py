"""Contrastive embedding CRNN, ported from Keras: magma mel -> ImageNet DenseNet121 -> frequency
folded into channels -> BiGRU -> attentive statistics pooling -> BN -> 256 -> 128-d L2-normalised
embedding.

Train with `training/unsupervised.py` (`experiments/legacy_simclr_crnn.py`). Correction over the Keras
script: there the two "views" were the same waveform passed twice through the model, differing only
by SpecAugment and dropout, so the task was nearly trivial; here the SSL trainer builds two
waveform-level augmented views (noise, speed, device EQ) first. Note the research finding that
same-recording positives must not pair complete with unfinished songs.

Origin: legacy `model/densenet_simclr.py` ("embed experiment").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attentive_stats_pool import AttentiveStatsPool
from layers.freq_to_channels import FreqToChannels
from layers.registry import build_frontend
from layers.temporal_bigru import TemporalBiGRU
from model.base import Model


class DenseNetCRNNEmbed(Model):
    """Colour-mapped mel spectrogram -> DenseNet121 -> BiGRU -> pooled, L2-normalised embedding.

    Args:
        n_classes: number of output classes of an optional linear classifier on the embedding;
            0 (default) builds none and `logits` is None.
        pretrained: True loads the torchvision ImageNet weights of DenseNet121 (a download on first
            use); False starts from random weights.
        frontend: dict of front-end settings (see `layers.registry.build_frontend`) that override
            this model's own defaults: 224-bin mel on 3-9 kHz with the 3-channel 'magma'
            normalisation.
        gru: channels of the frame series after the bidirectional GRU (each direction has half).
        emb_dim: size of the embedding.
        drop: dropout inside the projection MLP.

    `forward` returns a dict: `emb` (B, emb_dim) with unit length, `att` (B, T) frame weights,
    `frames` (B, gru, T) and `logits` (B, n_classes) or None.
    """

    backbone_prefixes = ('cnn.',)  # parameter-name prefixes that get `backbone_lr_factor`
    spec_fill = 'min'  # SpecAugment fills masked regions with the example minimum

    def __init__(self, n_classes=0, pretrained=True, frontend=None, gru=256, emb_dim=128, drop=0.3):
        super().__init__()
        import torchvision  # imported here, not at the top: only the legacy models need it

        # --- front-end: waveform -> mel spectrogram in dB, scaled to [0, 1], coloured with magma
        frontend_config = dict(bank='mel', norm='magma', n_bins=224, fmin=3000.0, fmax=9000.0)
        frontend_config.update(frontend or {})
        self.frontend = build_frontend(**frontend_config)

        # --- 2-D convolutional body: the convolutional part of torchvision's DenseNet121
        weights = 'DEFAULT' if pretrained else None
        self.cnn = torchvision.models.densenet121(weights=weights).features

        # --- ImageNet colour statistics the pretrained weights expect (constants, not saved)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', imagenet_mean, persistent=False)
        self.register_buffer('std', imagenet_std, persistent=False)

        # --- feature map -> frame series: fold the remaining frequency bins into channels
        # (DenseNet121 has 1024 output channels and downsamples frequency by 32)
        self.fold = FreqToChannels()
        freq_bins_left = self.frontend.n_bins // 32
        series_channels = 1024 * freq_bins_left

        # --- temporal module: bidirectional GRU over the frames
        self.gru = TemporalBiGRU(gru, d_in=series_channels)

        # --- frame series -> clip embedding
        self.pool = AttentiveStatsPool(gru)
        self.proj = nn.Sequential(
            nn.BatchNorm1d(2 * gru),
            nn.Linear(2 * gru, 256),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(256, emb_dim),
        )

        # --- optional classifier head on the embedding
        if n_classes:
            self.cls = nn.Linear(emb_dim, n_classes)
        else:
            self.cls = None

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 3, 224, 400)   RGB image, 10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        image = (spectrogram - self.mean) / self.std  # (B, 3, 224, 400)
        feature_map = self.cnn(image)  # (B, 1024, 7, 12)
        feature_map = F.relu(feature_map)  # (B, 1024, 7, 12)
        series = self.fold(feature_map)  # (B, 7168, 12)
        frames = self.gru(series)  # (B, 256, 12)
        pooled, frame_weights = self.pool(frames)  # (B, 512), (B, 12)
        projected = self.proj(pooled)  # (B, 128)
        embedding = F.normalize(projected, dim=1)  # (B, 128)   unit length

        if self.cls is not None:
            logits = self.cls(embedding)  # (B, n_classes)
        else:
            logits = None
        return {
            'emb': embedding,
            'att': frame_weights,
            'frames': frames,
            'logits': logits,
        }


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(DenseNetCRNNEmbed(pretrained=False))
