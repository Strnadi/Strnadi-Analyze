"""Broadcasted residual block (BC-ResNet, Kim et al. 2021; keyword-spotting / DCASE low-complexity models).

2-D frequency-depthwise features keep the frequency structure; a 1-D temporal branch runs on the
frequency-averaged signal and is broadcast back over frequency. Very few MACs, streaming-friendly.
Downsampling is along frequency only. Origin: fork review (architecture idea 9).
"""
import torch.nn as nn


class BCResBlock(nn.Module):
    def __init__(self, cin, cout, freq_stride=1, dilation=1, drop=0.1):
        super().__init__()
        self.transition = cin != cout or freq_stride != 1
        f2 = [nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)] if self.transition else []
        f2 += [nn.Conv2d(cout, cout, (3, 1), (freq_stride, 1), (1, 0), groups=cout, bias=False), nn.BatchNorm2d(cout)]
        self.f2 = nn.Sequential(*f2)
        self.f1 = nn.Sequential(nn.Conv2d(cout, cout, (1, 3), padding=(0, dilation), dilation=(1, dilation), groups=cout, bias=False),
                                nn.BatchNorm2d(cout), nn.SiLU(inplace=True), nn.Conv2d(cout, cout, 1, bias=False), nn.Dropout2d(drop))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):  # (B, C, F, T)
        x2 = self.f2(x)
        out = x2 + self.f1(x2.mean(2, keepdim=True))
        return self.act(out if self.transition else out + x)
