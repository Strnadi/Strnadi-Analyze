"""Sub-centre cosine classifier: logits = s * max_k cos(e, w_{c,k}) (Deng et al. 2020).

k > 1 lets a class own several prototypes (dialect variants, "mixed singers"). The margin is applied
by the loss (`losses/margin.py`), not here. Gives the embedding the angular geometry the BioAcu
"draw a circle" map needs. Origin: research session (`CosineLinear`).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubcenterCosineHead(nn.Module):
    def __init__(self, d, n_classes, k=1, s=16.0):
        super().__init__()
        self.k = k  # sub-centres per class
        self.s = s  # logit scale
        self.n = n_classes
        # (n_classes * k, D); rows c * k .. c * k + k - 1 are the sub-centres of class c
        self.weight = nn.Parameter(torch.randn(n_classes * k, d) * 0.01)
        self.in_features = d

    def forward(self, e):
        """Embedding e (B, D) -> {'logits': (B, n_classes)}."""
        embedding_unit = F.normalize(e, dim=1)
        centres_unit = F.normalize(self.weight, dim=1)
        # cosine of the embedding to every sub-centre
        cos = F.linear(embedding_unit, centres_unit)  # (B, n_classes * k)
        cos = cos.view(-1, self.n, self.k)  # (B, n_classes, k)
        best_cos = cos.amax(-1)  # (B, n_classes): the closest sub-centre of every class
        return {'logits': self.s * best_cos}
