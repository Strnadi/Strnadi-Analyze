"""Prototype head on the frame series: each class owns `k` prototypes; class score = best cosine match of any
prototype at any frame (max over time, mean over the class's prototypes).

The decision becomes "this stretch looks like prototype j of class c": interpretable (the matching frame is
returned), localised on the terminal phrase instead of pooled over the clip, and the right probe for frozen
foundation-model features (Perch 2.0 trains a prototype classifier; Bird-MAE needs prototypical probing;
AudioProtoPNet). Takes (B, D, T), not the pooled embedding.
Origin: research literature report + fork review (never built in the research session).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeHead(nn.Module):
    def __init__(self, d, n_classes, k=4, s=16.0):
        super().__init__()
        self.n, self.k, self.s = n_classes, k, s
        self.protos = nn.Parameter(torch.randn(n_classes * k, d) * 0.01)

    def forward(self, frames):  # (B, D, T)
        sim = torch.einsum('bdt,pd->bpt', F.normalize(frames, dim=1), F.normalize(self.protos, dim=1))
        best, when = sim.max(-1)                                      # (B, P)
        logits = self.s * best.view(-1, self.n, self.k).mean(-1)
        return {'logits': logits, 'proto_frame': when.view(-1, self.n, self.k)}
