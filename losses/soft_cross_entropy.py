"""Cross-entropy against hard or soft target distributions, with label smoothing and optional
focal modulation (Lin et al. 2017).

`logits` may be raw logits or log-probabilities (log-softmax of normalised log-probs is a no-op).

Returns (B,).
"""

import torch.nn.functional as F


def soft_cross_entropy(logits, target, smoothing=0.0, focal=0.0):
    """Cross-entropy of every example against a target distribution.

    Args:
        logits: (B, C) raw logits or log-probabilities.
        target: (B, C) target distribution per example: one-hot for a hard label, or soft.
        smoothing: label smoothing in 0 ... 1: this share of the target mass is spread evenly
            over all C classes.
        focal: focal exponent gamma; 0 switches the focal modulation off. With gamma > 0 a class
            the model already gives a high probability contributes less.

    Returns:
        (B,) loss per example (not averaged).
    """
    n_classes = target.shape[1]
    log_probs = F.log_softmax(logits, 1)  # (B, C)

    # Label smoothing: keep (1 - smoothing) of the target, spread the rest uniformly.
    smoothed_target = target * (1 - smoothing) + smoothing / n_classes  # (B, C)

    if focal > 0:
        # Focal modulation: scale each class term by (1 - p) ** gamma. The clamp keeps the factor
        # away from exactly 0 when p rounds to 1.
        probs = log_probs.exp()
        focal_factor = (1 - probs).clamp_min(1e-6) ** focal  # (B, C)
        log_probs = log_probs * focal_factor

    # -sum_c target_c * log p_c
    return -(smoothed_target * log_probs).sum(1)
