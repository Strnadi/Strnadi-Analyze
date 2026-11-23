"""Sub-centre cosine classifier: logits = s * max_k cos(e, w_{c,k}) (Deng et al. 2020).

k > 1 lets a class own several prototypes (dialect variants, "mixed singers"). The margin is applied by the
loss (`losses/margin.py`), not here. Gives the embedding the angular geometry the BioAcu "draw a circle" map
needs. Origin: research session (`CosineLinear`).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubcenterCosineHead(nn.Module):
    def __init__(self, d, n_classes, k=1, s=16.0):
        super().__init__()
        self.k, self.s, self.n = k, s, n_classes
        self.weight = nn.Parameter(torch.randn(n_classes * k, d) * 0.01); self.in_features = d

    def forward(self, e):
        cos = F.linear(F.normalize(e, dim=1), F.normalize(self.weight, dim=1)).view(-1, self.n, self.k).amax(-1)
        return {'logits': self.s * cos}
