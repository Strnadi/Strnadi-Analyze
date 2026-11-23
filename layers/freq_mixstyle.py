"""Freq-MixStyle (Kim et al. 2022; Schmid et al. DCASE 2022): mix per-frequency mean/std statistics between
examples of a batch. Frequency-wise statistics are mostly the recording device's response, so this simulates
unseen microphones (unseen-device accuracy 48 % -> 58 % in the cited DCASE study). Training only.
Origin: research literature report / backlog (never built).
"""
import torch
import torch.nn as nn


class FreqMixStyle(nn.Module):
    def __init__(self, p=0.4, alpha=0.4, eps=1e-6):
        super().__init__(); self.p, self.alpha, self.eps = p, alpha, eps

    def forward(self, x):  # (B, C, F, T)
        if not self.training or torch.rand(()) > self.p:
            return x
        B = x.shape[0]
        mu = x.mean(dim=(1, 3), keepdim=True); sd = (x.var(dim=(1, 3), keepdim=True) + self.eps).sqrt()
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x)
        perm = torch.randperm(B, device=x.device)
        mu2, sd2 = lam * mu + (1 - lam) * mu[perm], lam * sd + (1 - lam) * sd[perm]
        return (x - mu) / sd * sd2.detach() + mu2.detach()
