"""The production architecture ("good-model-2" lineage), ported from Keras: 224-bin mel (3-9 kHz) -> magma
colour map -> ImageNet DenseNet121 -> attentive statistics pooling over all patches -> BN -> 256 -> 64 -> classes.
8.2 M params; about 12x the training cost per step of StrnadNet on CPU.

Kept as the control for "is it the data or the architecture?". Differences from the Keras script that are
corrections, not changes of design:
- input is normalised with the ImageNet mean/std the pretrained weights expect (the Keras model fed raw [0, 1]);
- SpecAugment masks to the example minimum, per example (see layers/spec_augment.py);
- splits, augmentation and checkpointing come from the shared trainer (recording-grouped folds; the Keras
  script split randomly by file with a time-seeded split, so its reported validation scores were optimistic).
On held-out Vizina males the deployed model of this architecture calls ~45 % of complete BhBl songs
"Unfinished" (BhBl recall 0.55 over 58 males).
Origin: legacy `model/densenet.py`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.registry import build_frontend
from layers.attentive_stats_pool_2d import AttentiveStatsPool2D
from layers.mlp_head import MLPHead


class DenseNet121AttnPool(nn.Module):
    backbone_prefixes = ('cnn.',)
    spec_fill = 'min'

    def __init__(self, n_classes=7, pretrained=True, frontend=None):
        super().__init__()
        import torchvision
        fe = dict(bank='mel', norm='magma', n_bins=224, fmin=3000.0, fmax=9000.0)
        fe.update(frontend or {})
        self.frontend = build_frontend(**fe)
        self.cnn = torchvision.models.densenet121(weights='DEFAULT' if pretrained else None).features
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.pool = AttentiveStatsPool2D(1024)
        self.cls = MLPHead(2048, n_classes, hidden=(256, 64), drop=0.3)

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        f = F.relu(self.cnn((spec - self.mean) / self.std))
        e = self.pool(f)
        out = self.cls(e); out['emb'] = e
        return out
