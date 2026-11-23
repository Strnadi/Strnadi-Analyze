"""Additive margins for cosine heads (logits = s * cos): CosFace (cos - m) and ArcFace
(cos(theta + m)) on the target class.

Soft / multi-hot targets are supported for CosFace (margin scaled by the target weight).

Origin: research session (`--margin`), used with `SubcenterCosineHead` and the identity model.
"""

import math
import torch


def cosface_margin(logits, target, m=0.2, s=16.0):
    """Subtract the CosFace margin from the target-class logits.

    Args:
        logits: (B, C) logits of a cosine head, i.e. s * cos(angle to the class centre).
        target: (B, C) one-hot or soft target; the margin of a class is scaled by its target
            weight, so classes with weight 0 are left unchanged.
        m: margin in cosine units.
        s: the scale the cosine head multiplied the cosines by.

    Returns:
        (B, C) logits with s * m * target subtracted.
    """
    # The margin is m in cosine units, which is m * s in logit units.
    margin_in_logit_units = m * s
    return logits - margin_in_logit_units * target


def arcface_margin(logits, target_idx, m=0.3, s=16.0):
    """Add the ArcFace angular margin to the target-class angle.

    Args:
        logits: (B, C) logits of a cosine head, i.e. s * cos(angle to the class centre).
        target_idx: (B,) integer index of the target class of each example.
        m: margin in radians, added to the angle between the embedding and its target centre.
        s: the scale the cosine head multiplied the cosines by.

    Returns:
        (B, C) logits where the target class holds s * cos(theta + m) and the other classes
        s * cos(theta).
    """
    n_classes = logits.shape[1]

    # Back from logits to cosines; kept strictly inside (-1, 1) so acos and its gradient stay
    # finite.
    cos_theta = (logits / s).clamp(-1 + 1e-6, 1 - 1e-6)  # (B, C)
    theta = torch.acos(cos_theta)  # angle in radians, 0 ... pi

    # cos(theta + m) only keeps falling while theta + m < pi. Beyond that the usual ArcFace
    # fallback is used: a linear penalty cos - m * sin(m).
    cos_with_margin = torch.cos(theta + m)
    linear_fallback = cos_theta - m * math.sin(m)
    margin_cos = torch.where(theta + m < math.pi, cos_with_margin, linear_fallback)  # (B, C)

    # Use the margin version for the target class only.
    is_target = torch.nn.functional.one_hot(target_idx, n_classes).bool()  # (B, C)
    return s * torch.where(is_target, margin_cos, cos_theta)
