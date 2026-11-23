"""ArcFace head: cosine classifier with an additive angular margin on the target class (Deng et al. 2019).

`forward(e)` gives plain scaled cosines (inference); `forward(e, target)` adds the margin (training).
Origin: the legacy repo had an empty `layers/arcface.py` placeholder for this; the research identity model
(`training/train_identity.py`) uses the same margin through `losses/margin.py`.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, d, n_classes, s=30.0, m=0.3):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.randn(n_classes, d) * 0.01); self.in_features = d

    def forward(self, e, target=None):
        cos = F.linear(F.normalize(e, dim=1), F.normalize(self.weight, dim=1))
        if target is not None:
            theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            # keep the target logit monotonic when theta + m would pass pi
            marg = torch.where(theta + self.m < math.pi, torch.cos(theta + self.m), cos - self.m * math.sin(self.m))
            cos = torch.where(F.one_hot(target, cos.shape[1]).bool(), marg, cos)
        return {'logits': self.s * cos}
