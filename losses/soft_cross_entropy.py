"""Cross-entropy against hard or soft target distributions, with label smoothing and optional focal modulation
(Lin et al. 2017). `logits` may be raw logits or log-probabilities (log-softmax of normalised log-probs is a
no-op). Returns (B,)."""
import torch.nn.functional as F


def soft_cross_entropy(logits, target, smoothing=0.0, focal=0.0):
    logp = F.log_softmax(logits, 1)
    t = target * (1 - smoothing) + smoothing / target.shape[1]
    if focal > 0:
        logp = logp * (1 - logp.exp()).clamp_min(1e-6) ** focal
    return -(t * logp).sum(1)
