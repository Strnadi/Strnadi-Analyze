"""Generalised cross-entropy (Zhang & Sabuncu, 2018): a loss that tolerates some wrong labels.

    loss = (1 - p_label ** q) / q

For q -> 0 this is the usual cross-entropy, -log(p_label); for q = 1 it is the mean absolute error,
1 - p_label. In between, an example the model confidently contradicts (p_label near 0) contributes
a bounded loss and a small gradient, where cross-entropy lets it dominate the batch. That is the
wanted behaviour when 1-2 % of the labels are still wrong after the audit; the price is slower
learning from hard but correct examples. q = 0.7 is the value recommended by the authors.
"""

import torch


def generalized_cross_entropy(logits, targets, q=0.7):
    """Per-example loss.

    Args:
        logits: (B, C) logits, or log-probabilities (a log-softmax of them changes nothing).
        targets: (B, C) target vectors; the probability mass on the target classes is `p_label`
            (one-hot labels give the probability of the true class).
        q: 0 < q <= 1, see the module docstring.
    """
    probabilities = torch.softmax(logits, dim=1)
    p_label = (probabilities * targets).sum(1).clamp(1e-7, 1.0)
    return (1 - p_label.pow(q)) / q
