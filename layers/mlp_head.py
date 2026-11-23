"""BatchNorm -> [Linear -> SiLU -> Dropout] x n -> Linear classifier head.
Origin: legacy production heads (DenseNet: 256-64, MobileNet: 128, Perch: 512 with GELU)."""

import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, d, n_classes, hidden=(256, 64), drop=0.3, act=nn.SiLU, bn_in=True):
        super().__init__()
        layers = []
        if bn_in:
            layers.append(nn.BatchNorm1d(d))
        width = d  # width of the features entering the next layer
        for hidden_width in hidden:
            layers += [nn.Linear(width, hidden_width), act(), nn.Dropout(drop)]
            width = hidden_width
        self.body = nn.Sequential(*layers)
        self.lin = nn.Linear(width, n_classes)

    def forward(self, e):
        """Embedding e (B, D) -> {'logits': (B, n_classes)}."""
        hidden_features = self.body(e)  # (B, width of the last hidden layer)
        return {'logits': self.lin(hidden_features)}
