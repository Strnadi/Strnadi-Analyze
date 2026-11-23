"""Freq-MixStyle (Kim et al. 2022; Schmid et al. DCASE 2022): mix per-frequency mean/std statistics
between examples of a batch. Frequency-wise statistics are mostly the recording device's response,
so this simulates unseen microphones (unseen-device accuracy 48 % -> 58 % in the cited DCASE
study). Training only.
Origin: research literature report / backlog (never built).
"""

import torch
import torch.nn as nn


class FreqMixStyle(nn.Module):
    def __init__(self, p=0.4, alpha=0.4, eps=1e-6):
        super().__init__()
        self.p = p  # probability of applying the mix to a batch
        self.alpha = alpha  # Beta(alpha, alpha) parameter of the mixing coefficient
        self.eps = eps

    def forward(self, x):
        """(B, C, F, T) -> (B, C, F, T); returns x unchanged in eval mode and when not applied."""
        if not self.training or torch.rand(()) > self.p:
            return x
        B = x.shape[0]
        # statistics of every frequency bin, taken over channels and time
        mean = x.mean(dim=(1, 3), keepdim=True)  # (B, 1, F, 1)
        std = (x.var(dim=(1, 3), keepdim=True) + self.eps).sqrt()  # (B, 1, F, 1)

        # every example is mixed with a random partner from the batch
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((B, 1, 1, 1)).to(x)  # (B, 1, 1, 1) weight of the example's own statistics
        partner = torch.randperm(B, device=x.device)  # (B,) index of each example's partner
        mixed_mean = lam * mean + (1 - lam) * mean[partner]
        mixed_std = lam * std + (1 - lam) * std[partner]

        # remove the example's own statistics, put the mixed ones in (no gradient through them)
        standardised = (x - mean) / std
        return standardised * mixed_std.detach() + mixed_mean.detach()
