"""Frequency-dynamic convolution (Nam et al. 2022, the standard strong SED CRNN building block).

An ordinary conv is translation-equivariant along frequency, which is physically wrong for sounds
defined by absolute pitch. Here the kernel is a per-frequency-bin mixture of `n_basis` kernels, with
mixture weights predicted from the time-pooled input. Stride is (freq, time).
Origin: fork review (architecture idea 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqDynamicConv(nn.Module):
    def __init__(self, cin, cout, k=3, stride=(1, 1), n_basis=4, temperature=31.0, reduction=4):
        super().__init__()
        self.cout = cout
        self.k = k
        self.stride = tuple(stride)
        self.n_basis = n_basis
        self.temperature = temperature
        # all basis kernels stacked along the output-channel axis: (n_basis * cout, cin, k, k)
        self.weight = nn.Parameter(torch.empty(n_basis * cout, cin, k, k))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        # small 1-D network along frequency that predicts the mixture logits of every bin
        hidden = max(cin // reduction, 4)
        self.att = nn.Sequential(
            nn.Conv1d(cin, hidden, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, n_basis, 1),
        )

    def forward(self, x):
        """(B, cin, F, T) -> (B, cout, F', T') with F', T' set by `stride`."""
        time_pooled = x.mean(3)  # (B, cin, F)
        mixture_logits = self.att(time_pooled)  # (B, n_basis, F)
        # softmax over the basis kernels, separately for every frequency bin
        mixture_weights = torch.softmax(mixture_logits / self.temperature, 1)  # (B, n_basis, F)

        # run all basis kernels in one convolution
        basis_outputs = F.conv2d(x, self.weight, stride=self.stride, padding=self.k // 2)
        B, _, n_bins_out, n_frames_out = basis_outputs.shape  # (B, n_basis * cout, F', T')
        basis_outputs = basis_outputs.view(B, self.n_basis, self.cout, n_bins_out, n_frames_out)

        # bring the weights to the output frequency resolution, then mix the basis outputs
        mixture_weights = F.adaptive_avg_pool1d(mixture_weights, n_bins_out)  # (B, n_basis, F')
        mixture_weights = mixture_weights[:, :, None, :, None]  # (B, n_basis, 1, F', 1)
        return (basis_outputs * mixture_weights).sum(1)  # (B, cout, F', T')
