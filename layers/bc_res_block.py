"""Broadcasted residual block (BC-ResNet, Kim et al. 2021; keyword-spotting / DCASE low-complexity
models).

2-D frequency-depthwise features keep the frequency structure; a 1-D temporal branch runs on the
frequency-averaged signal and is broadcast back over frequency. Very few MACs, streaming-friendly.
Downsampling is along frequency only. Origin: fork review (architecture idea 9).
"""

import torch.nn as nn


class BCResBlock(nn.Module):
    def __init__(self, cin, cout, freq_stride=1, dilation=1, drop=0.1):
        super().__init__()
        # a transition block changes the channel count or the frequency resolution, so its input
        # cannot be added back as an identity skip
        self.transition = cin != cout or freq_stride != 1

        # f2: the 2-D branch, a depthwise (3 x 1) conv along frequency
        freq_layers = []
        if self.transition:
            # 1x1 conv to the new channel count first
            freq_layers += [
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            ]
        freq_depthwise = nn.Conv2d(
            cout,
            cout,
            kernel_size=(3, 1),
            stride=(freq_stride, 1),
            padding=(1, 0),
            groups=cout,
            bias=False,
        )
        freq_layers += [freq_depthwise, nn.BatchNorm2d(cout)]
        self.f2 = nn.Sequential(*freq_layers)

        # f1: the 1-D branch, a depthwise (1 x 3) conv along time on the frequency-averaged signal
        time_depthwise = nn.Conv2d(
            cout,
            cout,
            kernel_size=(1, 3),
            padding=(0, dilation),
            dilation=(1, dilation),
            groups=cout,
            bias=False,
        )
        self.f1 = nn.Sequential(
            time_depthwise,
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 1, bias=False),
            nn.Dropout2d(drop),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """(B, cin, F, T) -> (B, cout, F', T) with F' = ceil(F / freq_stride)."""
        freq_features = self.f2(x)  # (B, cout, F', T)
        freq_averaged = freq_features.mean(2, keepdim=True)  # (B, cout, 1, T)
        time_features = self.f1(freq_averaged)  # (B, cout, 1, T)
        out = freq_features + time_features  # the 1-D branch is broadcast over frequency
        if not self.transition:
            out = out + x  # identity skip
        return self.act(out)
