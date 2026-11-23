"""FDY-CRNN: frequency-dynamic conv stages + bidirectional GRU (Nam et al. 2022), the standard strong
sound-event-detection network, with the shared attentive head on top.

Frequency-dynamic kernels adapt along the frequency axis, which suits patterns defined by absolute pitch better
than one shared kernel plus a coordinate channel. Because it is an SED architecture its `frame_logits` are a
natural song detector. Status: proposed (fork review, architecture idea 4). Untrained.
"""
import torch.nn as nn
from layers.registry import build_frontend
from layers.freq_dynamic_conv import FreqDynamicConv
from layers.freq_to_channels import FreqToChannels
from layers.sequence_head import SequenceHead


class FDYCRNN(nn.Module):
    def __init__(self, n_classes=7, frontend=None, widths=(16, 32, 64, 128), d=160, drop=0.2, head='linear', head_kw=None):
        super().__init__()
        self.frontend = build_frontend(**(frontend or {}))
        layers, cin = [], self.frontend.out_channels
        for i, w in enumerate(widths):
            conv = nn.Conv2d(cin, w, 3, padding=1, bias=False) if i == 0 else FreqDynamicConv(cin, w)   # first layer static, as in the paper
            layers += [conv, nn.BatchNorm2d(w), nn.SiLU(inplace=True), nn.AvgPool2d((2, 2 if i < 2 else 1))]; cin = w
        self.cnn = nn.Sequential(*layers)
        self.fold = FreqToChannels()
        self.head = SequenceHead(cin * (self.frontend.n_bins // 2 ** len(widths)), n_classes, d, 'gru', 2, drop=drop, head=head, head_kw=head_kw)

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        return self.head(self.fold(self.cnn(spec)))
