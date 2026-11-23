"""Weak-label loss for the factorised head: a recording-level dialect label supervises ONLY q(dialect | complete),
weighted by the model's own detached belief that this song is a complete song.

With a flat 7-way candidate-set loss the cheapest way to satisfy "BC or Unfinished or None" is to call every
hard song Unfinished, which is exactly the field complaint. Routing the weak label to the dialect factor alone
removes that pull (R2: macro-F1 +0.029 [+0.013, +0.046] over the strong-only baseline; the same data without
this loss, R1, gave no significant gain).

Limitation (fork review): the weak songs then teach nothing about completeness, which is where almost all
remaining error is. `losses/bag_complete.py` adds that signal at recording level.
Returns a scalar. Origin: research session (reviewer idea N2).
"""
import torch

DIALECT_IDX = [0, 1, 2, 3, 6]


def factorized_weak_loss(out, candidates, weight):
    """out: model dict with log_song / log_complete / log_dialect; candidates (B, 7) mask (+ eps escape mass)."""
    cand = candidates[:, DIALECT_IDX]
    nll = -torch.logsumexp(out['log_dialect'] + torch.log(cand.clamp_min(1e-12)), 1)
    p_complete = (out['log_song'] + out['log_complete']).exp().detach()
    has_dialect = (cand >= 1).any(1).float()
    return (p_complete * has_dialect * nll * weight).sum() / weight.sum().clamp_min(1e-6)
