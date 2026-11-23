"""Stem block with long horizontal (time) kernels next to a 3x3 branch.

The terminal elements are thin, nearly horizontal lines lasting 0.2-0.5 s. A (1 x k) kernel sums energy along
such a line, which raises its contrast before any downsampling; a 3x3 kernel followed by stride 2 (the research
stem) sees only 3 frames of it. Branch outputs are concatenated. Stride is (freq, time).
Origin: fork review (architecture idea 2).
"""
import torch
import torch.nn as nn


class TonalLineConv(nn.Module):
    def __init__(self, cin, cout, time_kernels=(9, 15), stride=(1, 1)):
        super().__init__()
        n = len(time_kernels) + 1; widths = [cout - (cout // n) * (n - 1)] + [cout // n] * (n - 1)
        self.branches = nn.ModuleList([nn.Conv2d(cin, widths[0], 3, tuple(stride), 1, bias=False)])
        for k, w in zip(time_kernels, widths[1:]):
            self.branches.append(nn.Conv2d(cin, w, (1, k), tuple(stride), (0, k // 2), bias=False))
        self.bn = nn.BatchNorm2d(cout); self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(torch.cat([b(x) for b in self.branches], 1)))
