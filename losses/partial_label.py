"""Partial-label (candidate-set) negative log-likelihood: -log sum_{y in S} p(y).

A song from a recording filed as dialect D is labelled with the SET {D, Unfinished, None}:
recording labels say which dialect the bird has, not whether this particular song is complete.

`candidates` is a (B, C) mask with 1 for members; non-members may carry a small escape mass eps
(file labels are sometimes wrong, or another bird sings: 0-10 % of verified segments carry another
dialect).

Returns (B,).

Origin: research session (R1).
"""

import torch
import torch.nn.functional as F


def partial_label_nll(logits, candidates):
    """Negative log of the probability the model gives to the candidate set of each example.

    Args:
        logits: (B, C) raw logits or log-probabilities.
        candidates: (B, C) weights of the candidate set: 1 for a class that is in the set, 0 (or
            a small escape mass eps) for a class that is not.

    Returns:
        (B,) loss per example: -log sum_c candidates_c * p_c (not averaged).
    """
    log_probs = F.log_softmax(logits, 1)  # (B, C)

    # log of the candidate weights; the clamp turns log(0) = -inf into a large finite negative
    # number (a weight of 1e-12), so classes outside the set are effectively dropped from the sum.
    log_candidate_weights = torch.log(candidates.clamp_min(1e-12))  # (B, C)

    # log sum_c weight_c * p_c, computed in the log domain.
    log_p_candidates = torch.logsumexp(log_probs + log_candidate_weights, 1)  # (B,)
    return -log_p_candidates
