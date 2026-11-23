"""Linear projection of the student's clip embedding into a teacher's embedding space (e.g. Perch 2.0, 1536-d),
trained with `losses/distill_cosine.py`. This is the BirdCLEF 2026 winning recipe for small CPU models and it
gives every per-project head on the BioAcu platform one shared embedding space.
Origin: research session (`--w_distill`, implemented but never queued); both reviews rank it first.
"""
import torch.nn as nn


class DistillProjection(nn.Module):
    def __init__(self, d, teacher_dim=1536):
        super().__init__()
        self.proj = nn.Linear(d, teacher_dim)

    def forward(self, e):
        return self.proj(e)
