"""ArcFace head: cosine classifier with an additive angular margin on the target class
(Deng et al. 2019).

`forward(e)` gives plain scaled cosines (inference); `forward(e, target)` adds the margin
(training).
Origin: the legacy repo had an empty `layers/arcface.py` placeholder for this; the research
identity model (`training/identity.py`) uses the same margin through `losses/margin.py`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    def __init__(self, d, n_classes, s=30.0, m=0.3):
        super().__init__()
        self.s = s  # logit scale
        self.m = m  # angular margin in radians
        self.weight = nn.Parameter(torch.randn(n_classes, d) * 0.01)  # (n_classes, D) class centres
        self.in_features = d

    def forward(self, e, target=None):
        """Embedding e (B, D), optional target (B,) class indices -> {'logits': (B, n_classes)}."""
        embedding_unit = F.normalize(e, dim=1)
        centres_unit = F.normalize(self.weight, dim=1)
        cos = F.linear(embedding_unit, centres_unit)  # (B, n_classes) cosine to every class centre
        if target is not None:
            # angle between the embedding and every class centre
            theta = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            # keep the target logit monotonic when theta + m would pass pi
            cos_with_margin = torch.where(
                theta + self.m < math.pi,
                torch.cos(theta + self.m),
                cos - self.m * math.sin(self.m),
            )
            is_target = F.one_hot(target, cos.shape[1]).bool()  # (B, n_classes)
            # the margin is applied to the target class only
            cos = torch.where(is_target, cos_with_margin, cos)
        return {'logits': self.s * cos}
