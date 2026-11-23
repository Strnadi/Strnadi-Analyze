"""Gated-attention multiple-instance pooling (Ilse et al. 2018) over a bag of song embeddings.

(B, N, D) with mask (B, N) -> bag embedding (B, D) and instance weights (B, N). Used to pool all
songs of one recording, so a recording-level dialect label becomes correct bag supervision (not a
noisy per-song label).
Origin: fork review (architecture idea 8); the research backlog listed attention-MIL but never
built it.
"""

import torch
import torch.nn as nn


class AttentionMILPool(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        # the short attribute names are state_dict keys of trained checkpoints
        self.v = nn.Linear(d, hidden)  # tanh branch
        self.u = nn.Linear(d, hidden)  # sigmoid gate branch
        self.w = nn.Linear(hidden, 1)  # one attention score per instance

    def forward(self, x, mask=None):
        """Bag x (B, N, D), optional mask (B, N) with True = real instance -> bag embedding (B, D),
        instance weights (B, N)."""
        tanh_branch = torch.tanh(self.v(x))  # (B, N, hidden)
        gate = torch.sigmoid(self.u(x))  # (B, N, hidden)
        scores = self.w(tanh_branch * gate).squeeze(-1)  # (B, N)
        if mask is not None:
            # masked-out (padding) instances get weight 0 after the softmax
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        attention_weights = torch.softmax(scores, 1)  # (B, N), sums to 1 over the bag
        bag_embedding = (x * attention_weights.unsqueeze(-1)).sum(1)  # (B, D)
        return bag_embedding, attention_weights
