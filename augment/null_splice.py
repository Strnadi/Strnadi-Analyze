"""Cut the clip at k random points and re-join it with the same 5 ms cross-fade the counterfactual surgery uses.

The synthetic (swapped / truncated) songs contain such joins; if only they had them, "has a join" would be a
shortcut to their labels. Applying inaudible joins to every class removes it. Origin: research session
(independent-review fix V5)."""
import numpy as np
from .fill import fill_to


def null_splice(y, k=None, fade=160):
    k = k or np.random.randint(1, 5)
    n = len(y); pts = sorted(np.random.randint(int(0.2 * n), int(0.95 * n), k))
    parts, last = [], 0
    for p in pts:
        parts.append(y[last:p]); last = p
    parts.append(y[last:]); out = parts[0].copy()
    w = np.linspace(0, 1, fade, dtype=np.float32)
    for q in parts[1:]:
        if len(out) >= fade and len(q) >= fade:
            out[-fade:] = out[-fade:] * (1 - w) + q[:fade] * w; out = np.concatenate([out, q[fade:]])
        else:
            out = np.concatenate([out, q])
    return fill_to(out, n)
