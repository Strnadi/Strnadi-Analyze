"""Additive margins for cosine heads (logits = s * cos): CosFace (cos - m) and ArcFace (cos(theta + m)) on the
target class. Soft / multi-hot targets are supported for CosFace (margin scaled by the target weight).
Origin: research session (`--margin`), used with `SubcenterCosineHead` and the identity model."""
import math
import torch


def cosface_margin(logits, target, m=0.2, s=16.0):
    """target: (B, C) one-hot or soft."""
    return logits - m * s * target


def arcface_margin(logits, target_idx, m=0.3, s=16.0):
    cos = (logits / s).clamp(-1 + 1e-6, 1 - 1e-6); theta = torch.acos(cos)
    marg = torch.where(theta + m < math.pi, torch.cos(theta + m), cos - m * math.sin(m))
    return s * torch.where(torch.nn.functional.one_hot(target_idx, cos.shape[1]).bool(), marg, cos)
