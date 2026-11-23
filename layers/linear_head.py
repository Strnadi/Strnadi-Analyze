"""Plain linear classifier head on the clip embedding. Every head returns a dict with at least
`logits`."""

import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, d, n_classes):
        super().__init__()
        self.lin = nn.Linear(d, n_classes)
        self.in_features = d

    def forward(self, e):
        """Embedding e (B, D) -> {'logits': (B, n_classes)}."""
        return {'logits': self.lin(e)}
