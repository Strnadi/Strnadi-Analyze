"""Stem block with long horizontal (time) kernels next to a 3x3 branch.

The terminal elements are thin, nearly horizontal lines lasting 0.2-0.5 s. A (1 x k) kernel sums
energy along such a line, which raises its contrast before any downsampling; a 3x3 kernel followed
by stride 2 (the research stem) sees only 3 frames of it. Branch outputs are concatenated. Stride
is (freq, time).
Origin: fork review (architecture idea 2).
"""

import torch
import torch.nn as nn


class TonalLineConv(nn.Module):
    def __init__(self, cin, cout, time_kernels=(9, 15), stride=(1, 1)):
        super().__init__()
        stride = tuple(stride)
        # cout is split evenly over the branches; the 3x3 branch also takes the remainder
        n_branches = len(time_kernels) + 1
        line_channels = cout // n_branches
        square_channels = cout - line_channels * (n_branches - 1)

        square_branch = nn.Conv2d(cin, square_channels, 3, stride, 1, bias=False)
        self.branches = nn.ModuleList([square_branch])
        for kernel_length in time_kernels:
            line_branch = nn.Conv2d(
                cin,
                line_channels,
                kernel_size=(1, kernel_length),
                stride=stride,
                padding=(0, kernel_length // 2),
                bias=False,
            )
            self.branches.append(line_branch)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        """(B, cin, F, T) -> (B, cout, F', T') with F', T' set by `stride`."""
        branch_outputs = [branch(x) for branch in self.branches]
        stacked = torch.cat(branch_outputs, 1)  # (B, cout, F', T')
        return self.act(self.bn(stacked))
