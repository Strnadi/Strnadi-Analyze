"""Partial-label (candidate-set) negative log-likelihood: -log sum_{y in S} p(y).

A song from a recording filed as dialect D is labelled with the SET {D, Unfinished, None}: recording labels say
which dialect the bird has, not whether this particular song is complete. `candidates` is a (B, C) mask with 1
for members; non-members may carry a small escape mass eps (file labels are sometimes wrong, or another bird
sings: 0-10 % of verified segments carry another dialect). Returns (B,).
Origin: research session (R1).
"""
import torch
import torch.nn.functional as F


def partial_label_nll(logits, candidates):
    logp = F.log_softmax(logits, 1)
    return -torch.logsumexp(logp + torch.log(candidates.clamp_min(1e-12)), 1)
