"""BC-ResNet (Kim et al. 2021) adapted to 4 s song windows: a tiny, streaming-friendly model family
(tau = 1: ~0.05 M params before the head; tau = 3: ~0.3 M) proven in keyword spotting and the DCASE
low-complexity tasks.

It answers "how small can the phone model get": StrnadNet is a home-made design in the same spirit (frequency
structure kept, 1-D temporal processing) but was never compared against the published family.
The original ends in global pooling; here the trunk output goes through the shared sequence head, because
element order has to stay visible. Status: proposed (fork review, architecture idea 9). Untrained.
"""
import torch.nn as nn
from layers.registry import build_frontend
from layers.bc_res_block import BCResBlock
from layers.freq_to_channels import FreqToChannels
from layers.sequence_head import SequenceHead


class BCResNet(nn.Module):
    def __init__(self, n_classes=7, tau=2.0, frontend=None, d=96, temporal='conv', n_temporal=2, drop=0.1,
                 head='linear', head_kw=None):
        super().__init__()
        self.frontend = build_frontend(**(frontend or {}))
        c = [int(tau * w) for w in (16, 8, 12, 16, 20)]
        layers = [nn.Conv2d(self.frontend.out_channels, c[0], 5, (2, 2), 2, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True)]
        cin = c[0]
        for cout, n, fs, dil in ((c[1], 2, 1, 1), (c[2], 2, 2, 2), (c[3], 4, 2, 4), (c[4], 4, 2, 8)):
            for j in range(n):
                layers.append(BCResBlock(cin, cout, fs if j == 0 else 1, dil, drop)); cin = cout
        layers.append(nn.AvgPool2d((1, 2)))                              # 20 ms -> 40 ms frames
        self.cnn = nn.Sequential(*layers)
        self.fold = FreqToChannels()
        self.head = SequenceHead(cin * (self.frontend.n_bins // 16), n_classes, d, temporal, n_temporal, drop=drop, head=head, head_kw=head_kw)

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        return self.head(self.fold(self.cnn(spec)))
