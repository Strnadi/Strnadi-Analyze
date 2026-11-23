"""Supervised contrastive loss (Khosla et al. 2020): items sharing a label are positives.

With label = song id it is "same song under two augmentations"; with label = ring-coded male it
trains the individual-identity embedding.

Do NOT use recording id as the label for dialect work: it pulls complete and unfinished songs of
one bird together, the very distinction the classifier has to make (independent-review finding
V7).

Origin: research session (`ssl_pretrain.py`, now `training/pretrain.py`).
"""

import torch
import torch.nn.functional as F


def supcon(z, labels, tau=0.1):
    """Supervised contrastive loss over one batch of embeddings.

    Args:
        z: (B, D) embeddings (L2-normalised here).
        labels: (B,) labels; rows with equal labels are positives of each other.
        tau: temperature that divides the cosine similarities.

    Returns:
        Scalar: for every row that has at least one positive, minus the mean log-probability of
        its positives; averaged over those rows. Zero (still connected to `z` for autograd) when
        no row has a positive.
    """
    batch_size = len(z)
    z = F.normalize(z, dim=1)

    # Cosine similarity of every pair, as logits.
    similarity = z @ z.t() / tau  # (B, B)

    is_self = torch.eye(batch_size, dtype=torch.bool, device=z.device)  # (B, B)
    same_label = labels[:, None] == labels[None]  # (B, B)
    is_positive = same_label & ~is_self

    # A row must not be compared with itself. A large negative number is used instead of -inf,
    # because -inf * 0 in the masked sum below would give NaN.
    similarity = similarity.masked_fill(is_self, -1e9)

    # log-softmax over each row: log-probability of picking each other item.
    log_prob = similarity - torch.logsumexp(similarity, 1, keepdim=True)  # (B, B)

    has_positive = is_positive.any(1)  # (B,)
    if not has_positive.any():
        return z.sum() * 0

    # Mean log-probability of the positives of each row, for the rows that have any.
    sum_log_prob_positives = (log_prob * is_positive).sum(1)[has_positive]
    n_positives = is_positive.sum(1)[has_positive]
    mean_log_prob_positives = sum_log_prob_positives / n_positives
    return -mean_log_prob_positives.mean()
