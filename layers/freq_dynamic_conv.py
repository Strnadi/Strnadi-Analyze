"""Frequency-dynamic convolution (Nam et al. 2022, the standard strong SED CRNN building block).

An ordinary conv is translation-equivariant along frequency, which is physically wrong for sounds defined by
absolute pitch. Here the kernel is a per-frequency-bin mixture of `n_basis` kernels, with mixture weights
predicted from the time-pooled input. Stride is (freq, time). Origin: fork review (architecture idea 4).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqDynamicConv(nn.Module):
    def __init__(self, cin, cout, k=3, stride=(1, 1), n_basis=4, temperature=31.0, reduction=4):
        super().__init__()
        self.cout, self.k, self.stride, self.n_basis, self.temperature = cout, k, tuple(stride), n_basis, temperature
        self.weight = nn.Parameter(torch.empty(n_basis * cout, cin, k, k))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        hidden = max(cin // reduction, 4)
        self.att = nn.Sequential(nn.Conv1d(cin, hidden, 3, padding=1, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                                 nn.Conv1d(hidden, n_basis, 1))

    def forward(self, x):  # (B, C, F, T)
        a = torch.softmax(self.att(x.mean(3)) / self.temperature, 1)             # (B, K, F)
        y = F.conv2d(x, self.weight, stride=self.stride, padding=self.k // 2)    # (B, K*cout, F', T')
        B, _, Fo, To = y.shape
        a = F.adaptive_avg_pool1d(a, Fo)
        return (y.view(B, self.n_basis, self.cout, Fo, To) * a[:, :, None, :, None]).sum(1)
