"""DenseNet built from scratch with configurable block counts (e.g. [4, 8, 16, 12] or [2, 4, 8, 6]) on a
1-channel spectrogram, with the shared sequence head.

Origin: legacy `layers/densenet.py` (a copy of the Keras DenseNet with custom `blocks`, commented out in
`model/densenet.py`). The legacy version pooled over patches; this one keeps the time axis like every other
model here, because pooled patches cannot represent element order.
"""
import torch.nn as nn
from layers.registry import build_frontend
from layers.dense_block import DenseBlock
from layers.dense_transition import DenseTransition
from layers.freq_to_channels import FreqToChannels
from layers.sequence_head import SequenceHead


class DenseNetSmall(nn.Module):
    def __init__(self, n_classes=7, blocks=(4, 8, 16, 12), growth=32, frontend=None, d=160, temporal='conv', head='linear', head_kw=None):
        super().__init__()
        self.frontend = build_frontend(**(frontend or {}))
        c = 2 * growth
        layers = [nn.Conv2d(self.frontend.out_channels, c, 7, 2, 3, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
        for i, n in enumerate(blocks):
            blk = DenseBlock(c, n, growth); layers.append(blk); c = blk.out_channels
            if i < len(blocks) - 1:
                tr = DenseTransition(c); layers.append(tr); c = tr.out_channels
        layers += [nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
        self.cnn = nn.Sequential(*layers)
        self.fold = FreqToChannels()
        fdim = self.frontend.n_bins // (2 ** len(blocks))
        self.head = SequenceHead(c * fdim, n_classes, d, temporal, head=head, head_kw=head_kw)

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        return self.head(self.fold(self.cnn(spec)))
