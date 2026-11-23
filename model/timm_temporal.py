"""Any timm CNN as the 2-D backbone, with the shared sequence head on its last feature map.

The feature map keeps its time axis (output stride 16 where the architecture allows it); remaining frequency
bins are folded into channels. Used for the architecture comparison (ConvNeXt-atto, MobileNetV3, RegNetY,
EfficientNet-B0, ResNet18, DenseNet121 as the "same data, production architecture" control).

Fairness note (base-session review): pretrained backbones need a lower learning rate than the from-scratch
head. Train with `backbone_lr_mult` (e.g. 0.1) in the experiment config; the research sweep used the
from-scratch settings for everything, so its ranking mostly reflected hyper-parameters.
Origin: research session (`TimmNet`).
"""
import torch
import torch.nn as nn
from layers.registry import build_frontend
from layers.freq_to_channels import FreqToChannels
from layers.sequence_head import SequenceHead


class TimmTemporal(nn.Module):
    backbone_prefixes = ('enc.',)   # parameter-name prefixes that get `backbone_lr_mult`

    def __init__(self, n_classes=7, backbone='convnext_atto', pretrained=True, frontend=None, d=192, temporal='conv',
                 n_temporal=3, drop=0.2, output_stride=16, head='linear', head_kw=None):
        super().__init__()
        import timm
        self.frontend = build_frontend(**(frontend or {}))
        kw = dict(pretrained=pretrained, in_chans=self.frontend.out_channels, features_only=True, out_indices=(-1,))
        try:
            self.enc = timm.create_model(backbone, output_stride=output_stride, **kw)
        except Exception:
            self.enc = timm.create_model(backbone, **kw)
        self.fold = FreqToChannels()
        with torch.no_grad():
            f = self.enc(torch.zeros(1, self.frontend.out_channels, self.frontend.n_bins, 400))[-1]
        self.head = SequenceHead(f.shape[1] * f.shape[2], n_classes, d, temporal, n_temporal, drop=drop, head=head, head_kw=head_kw)

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        return self.head(self.fold(self.enc(spec)[-1]))
