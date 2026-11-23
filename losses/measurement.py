"""Auxiliary regression of the expert song-ending measurements (for `layers/measurement_head.py`).

`target` is (B, 8): f1, f2, f3, df, bw (kHz), t2 (s), has_first, has_second; NaN = not measured for this clip.
Smooth-L1 on the six continuous values, BCE on the two presence flags, each over the entries that exist.
Returns a scalar."""
import torch
import torch.nn.functional as F


def measurement_loss(pred, target):
    known = ~torch.isnan(target)
    if not known.any():
        return pred.sum() * 0
    t = torch.nan_to_num(target)
    cont = F.smooth_l1_loss(pred[:, :6], t[:, :6], reduction='none')[known[:, :6]]
    pres = F.binary_cross_entropy_with_logits(pred[:, 6:], t[:, 6:], reduction='none')[known[:, 6:]]
    parts = [x.mean() for x in (cont, pres) if x.numel()]
    return sum(parts) / len(parts)
