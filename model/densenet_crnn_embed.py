"""Contrastive embedding CRNN, ported from Keras: magma mel -> ImageNet DenseNet121 -> frequency folded into
channels -> BiGRU -> attentive statistics pooling -> BN -> 256 -> 128-d L2-normalised embedding.

Train with `python -m training.ssl_pretrain` / `losses/nt_xent.py`. Correction over the Keras script: there the
two "views" were the same waveform passed twice through the model, differing only by SpecAugment and dropout,
so the task was nearly trivial; here the SSL trainer builds two waveform-level augmented views (noise, speed,
device EQ) first. Note the research finding that same-recording positives must not pair complete with
unfinished songs.
Origin: legacy `model/densenet_simclr.py` ("embed experiment").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.registry import build_frontend
from layers.freq_to_channels import FreqToChannels
from layers.temporal_bigru import TemporalBiGRU
from layers.attentive_stats_pool import AttentiveStatsPool


class DenseNetCRNNEmbed(nn.Module):
    backbone_prefixes = ('cnn.',)
    spec_fill = 'min'

    def __init__(self, n_classes=0, pretrained=True, frontend=None, gru=256, emb_dim=128, drop=0.3):
        super().__init__()
        import torchvision
        fe = dict(bank='mel', norm='magma', n_bins=224, fmin=3000.0, fmax=9000.0)
        fe.update(frontend or {})
        self.frontend = build_frontend(**fe)
        self.cnn = torchvision.models.densenet121(weights='DEFAULT' if pretrained else None).features
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.fold = FreqToChannels()
        self.gru = TemporalBiGRU(gru, d_in=1024 * (self.frontend.n_bins // 32))
        self.pool = AttentiveStatsPool(gru)
        self.proj = nn.Sequential(nn.BatchNorm1d(2 * gru), nn.Linear(2 * gru, 256), nn.SiLU(), nn.Dropout(drop), nn.Linear(256, emb_dim))
        self.cls = nn.Linear(emb_dim, n_classes) if n_classes else None

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        h = self.gru(self.fold(F.relu(self.cnn((spec - self.mean) / self.std))))
        pooled, att = self.pool(h)
        e = F.normalize(self.proj(pooled), dim=1)
        return {'emb': e, 'att': att, 'frames': h, 'logits': self.cls(e) if self.cls is not None else None}
