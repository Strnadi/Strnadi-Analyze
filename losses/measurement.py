"""Auxiliary regression of the expert song-ending measurements (for `layers/measurement_head.py`).

`target` is (B, 8): f1, f2, f3, df, bw (kHz), t2 (s), has_first, has_second; NaN = not measured
for this clip.

Smooth-L1 on the six continuous values, BCE on the two presence flags, each over the entries that
exist.

Returns a scalar.
"""

import torch
import torch.nn.functional as F

# Columns 0-5 are the continuous measurements (f1, f2, f3, df, bw, t2); columns 6-7 are the
# presence flags (has_first, has_second).
_N_CONTINUOUS = 6


def measurement_loss(pred, target):
    """Loss between predicted and expert-measured song-ending measurements.

    Args:
        pred: (B, 8) predictions: six continuous values, then two logits for the presence flags.
        target: (B, 8) measurements in the same column order; NaN where a value was not measured.

    Returns:
        Scalar: the average of the two part losses (smooth-L1 over the measured continuous
        entries, BCE over the measured flags); a part without any measured entry is left out.
        Zero (still connected to `pred` for autograd) when nothing at all was measured.
    """
    is_known = ~torch.isnan(target)  # (B, 8)
    if not is_known.any():
        return pred.sum() * 0

    # Replace NaN by 0 so the element-wise losses stay finite; those entries are dropped below.
    target_filled = torch.nan_to_num(target)

    # Continuous measurements: smooth-L1 on the measured entries.
    continuous_errors = F.smooth_l1_loss(
        pred[:, :_N_CONTINUOUS],
        target_filled[:, :_N_CONTINUOUS],
        reduction='none',
    )  # (B, 6)
    continuous_errors = continuous_errors[is_known[:, :_N_CONTINUOUS]]  # 1-D, measured only

    # Presence flags: binary cross-entropy on the measured entries.
    presence_errors = F.binary_cross_entropy_with_logits(
        pred[:, _N_CONTINUOUS:],
        target_filled[:, _N_CONTINUOUS:],
        reduction='none',
    )  # (B, 2)
    presence_errors = presence_errors[is_known[:, _N_CONTINUOUS:]]  # 1-D, measured only

    # Average the two parts, skipping a part that has no measured entry in this batch.
    part_losses = []
    for errors in (continuous_errors, presence_errors):
        if errors.numel() > 0:
            part_losses.append(errors.mean())
    return sum(part_losses) / len(part_losses)
