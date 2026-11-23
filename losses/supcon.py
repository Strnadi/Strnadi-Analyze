"""Supervised contrastive loss (Khosla et al. 2020): items sharing a label are positives.

With label = song id it is "same song under two augmentations"; with label = ring-coded male it trains the
individual-identity embedding. Do NOT use recording id as the label for dialect work: it pulls complete and
unfinished songs of one bird together, the very distinction the classifier has to make (independent-review
finding V7). Origin: research session (`ssl_pretrain.py::supcon`).
"""
import torch
import torch.nn.functional as F


def supcon(z, labels, tau=0.1):
    z = F.normalize(z, dim=1); sim = z @ z.t() / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None]) & ~eye
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    has = pos.any(1)
    if not has.any():
        return z.sum() * 0
    return -((logp * pos).sum(1)[has] / pos.sum(1)[has]).mean()
