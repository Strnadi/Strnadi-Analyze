"""Prototype head on the frame series: each class owns `k` prototypes; class score = best cosine
match of any prototype at any frame (max over time, mean over the class's prototypes).

The decision becomes "this stretch looks like prototype j of class c": interpretable (the matching
frame is returned), localised on the terminal phrase instead of pooled over the clip, and the right
probe for frozen foundation-model features (Perch 2.0 trains a prototype classifier; Bird-MAE needs
prototypical probing; AudioProtoPNet). Takes (B, D, T), not the pooled embedding.
Origin: research literature report + fork review (never built in the research session).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):
    def __init__(self, d, n_classes, k=4, s=16.0):
        super().__init__()
        self.n = n_classes
        self.k = k  # prototypes per class
        self.s = s  # logit scale
        # (P, D) with P = n_classes * k; rows c * k .. c * k + k - 1 belong to class c
        self.protos = nn.Parameter(torch.randn(n_classes * k, d) * 0.01)

    def forward(self, frames):
        """Frame series (B, D, T) -> dict: `logits` (B, n_classes), `proto_frame` (B, n_classes, k)
        = index of the frame that matches each prototype best."""
        frames_unit = F.normalize(frames, dim=1)  # unit length along D
        protos_unit = F.normalize(self.protos, dim=1)
        # cosine similarity of every prototype with every frame
        similarity = torch.einsum('bdt,pd->bpt', frames_unit, protos_unit)  # (B, P, T)
        best_similarity, best_frame = similarity.max(-1)  # both (B, P): max over time
        best_per_class = best_similarity.view(-1, self.n, self.k)  # (B, n_classes, k)
        logits = self.s * best_per_class.mean(-1)  # (B, n_classes): mean over a class's prototypes
        return {'logits': logits, 'proto_frame': best_frame.view(-1, self.n, self.k)}
