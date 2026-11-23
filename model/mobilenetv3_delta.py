"""Low-latency legacy variant, ported from Keras: 224-bin mel (3-9 kHz) -> [dB, delta, delta-delta] channels
scaled to [0, 1] -> ImageNet MobileNetV3-Small -> attentive statistics pooling -> BN -> 128 -> classes.

Same corrections as `densenet121_attnpool.py` (ImageNet normalisation, per-example SpecAugment with min fill,
grouped folds via the shared trainer). Origin: legacy `model/mobilenet.py`.
"""
import torch
import torch.nn as nn
from layers.registry import build_frontend
from layers.attentive_stats_pool_2d import AttentiveStatsPool2D
from layers.mlp_head import MLPHead


class MobileNetV3Delta(nn.Module):
    backbone_prefixes = ('cnn.',)
    spec_fill = 'min'

    def __init__(self, n_classes=7, pretrained=True, frontend=None):
        super().__init__()
        import torchvision
        fe = dict(bank='mel', norm='delta', n_bins=224, fmin=3000.0, fmax=9000.0)
        fe.update(frontend or {})
        self.frontend = build_frontend(**fe)
        self.cnn = torchvision.models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None).features
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.pool = AttentiveStatsPool2D(576)
        self.cls = MLPHead(1152, n_classes, hidden=(128,), drop=0.3)

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        e = self.pool(self.cnn((spec - self.mean) / self.std))
        out = self.cls(e); out['emb'] = e
        return out
