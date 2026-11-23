"""Weak-label loss for the factorised head: a recording-level dialect label supervises ONLY
q(dialect | complete), weighted by the model's own detached belief that this song is a complete
song.

With a flat 7-way candidate-set loss the cheapest way to satisfy "BC or Unfinished or None" is to
call every hard song Unfinished, which is exactly the field complaint. Routing the weak label to
the dialect factor alone removes that pull (R2: macro-F1 +0.029 [+0.013, +0.046] over the
strong-only baseline; the same data without this loss, R1, gave no significant gain).

Limitation (fork review): the weak songs then teach nothing about completeness, which is where
almost all remaining error is. `losses/bag_complete.py` adds that signal at recording level.

Returns a scalar.

Origin: research session (reviewer idea N2).
"""

import torch

# Positions of the 5 dialects among the 7 classes; the other two (4, 5) are None and Unfinished.
DIALECT_IDX = [0, 1, 2, 3, 6]


def factorized_weak_loss(out, candidates, weight):
    """Candidate-set loss on the dialect factor only, weighted by the belief "complete song".

    Args:
        out: model output dict of the factorised head with
            `log_song` (B,) log p(song),
            `log_complete` (B,) log p(complete | song),
            `log_dialect` (B, 5) log q(dialect | complete), in the order of `DIALECT_IDX`.
        candidates: (B, 7) candidate-set mask over the 7 classes: 1 for members, 0 or a small
            escape mass eps for the others.
        weight: (B,) example weights.

    Returns:
        Scalar: the weighted sum of the per-example losses divided by the sum of the weights.
    """
    # Keep only the dialect columns of the candidate mask.
    dialect_candidates = candidates[:, DIALECT_IDX]  # (B, 5)

    # Candidate-set NLL on the dialect factor: -log sum_{d in set} q(d | complete).
    # The clamp turns log(0) into a large finite negative number (see partial_label.py).
    log_candidate_weights = torch.log(dialect_candidates.clamp_min(1e-12))  # (B, 5)
    log_q_candidates = torch.logsumexp(out['log_dialect'] + log_candidate_weights, 1)  # (B,)
    dialect_nll = -log_q_candidates

    # The model's belief that this is a complete song: p(song) * p(complete | song). Detached, so
    # the weak label cannot change it (it only scales the loss).
    p_complete = (out['log_song'] + out['log_complete']).exp().detach()  # (B,)

    # 1 for examples whose candidate set names at least one dialect, 0 for the rest (they have
    # nothing to teach the dialect factor).
    has_dialect = (dialect_candidates >= 1).any(1).float()  # (B,)

    weighted_nll = p_complete * has_dialect * dialect_nll * weight  # (B,)
    total_weight = weight.sum().clamp_min(1e-6)
    return weighted_nll.sum() / total_weight
