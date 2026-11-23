"""Gated-attention multiple-instance pooling (Ilse et al. 2018) over a bag of song embeddings.

(B, N, D) with mask (B, N) -> bag embedding (B, D) and instance weights (B, N). Used to pool all songs of one
recording, so a recording-level dialect label becomes correct bag supervision (not a noisy per-song label).
Origin: fork review (architecture idea 8); the research backlog listed attention-MIL but never built it.
"""
import torch
import torch.nn as nn


class AttentionMILPool(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(d, hidden), nn.Linear(d, hidden), nn.Linear(hidden, 1)

    def forward(self, x, mask=None):
        s = self.w(torch.tanh(self.v(x)) * torch.sigmoid(self.u(x))).squeeze(-1)   # (B, N)
        if mask is not None:
            s = s.masked_fill(~mask.bool(), float('-inf'))
        a = torch.softmax(s, 1)
        return (x * a.unsqueeze(-1)).sum(1), a
