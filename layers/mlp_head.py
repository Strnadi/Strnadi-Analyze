"""BatchNorm -> [Linear -> SiLU -> Dropout] x n -> Linear classifier head.
Origin: legacy production heads (DenseNet: 256-64, MobileNet: 128, Perch: 512 with GELU)."""
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, d, n_classes, hidden=(256, 64), drop=0.3, act=nn.SiLU, bn_in=True):
        super().__init__()
        layers = [nn.BatchNorm1d(d)] if bn_in else []
        for h in hidden:
            layers += [nn.Linear(d, h), act(), nn.Dropout(drop)]; d = h
        self.body = nn.Sequential(*layers); self.lin = nn.Linear(d, n_classes)

    def forward(self, e):
        return {'logits': self.lin(self.body(e))}
