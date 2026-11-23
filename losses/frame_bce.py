"""Frame-level song-activity loss: binary cross-entropy between per-frame logits (B, T) and 0/1
targets given at any frame rate (resampled to T).

Target frames < 0 are ignored (clips without a known song span).

Used to train `model/song_detector.py` and the `frame_logits` of every sequence model.

Returns a scalar.
"""

import torch.nn.functional as F


def frame_bce(frame_logits, target):
    """Binary cross-entropy between per-frame song-activity logits and targets.

    Args:
        frame_logits: (B, T) logits, one per time frame.
        target: (B, T') float targets: 1 = song, 0 = no song, negative = unknown (ignored). T' may
            differ from T; the targets are then resampled to T by nearest neighbour.

    Returns:
        Scalar: mean loss over the frames with a known target. Zero (still connected to
        `frame_logits` for autograd) when no frame has one.
    """
    n_frames = frame_logits.shape[-1]

    if target.shape[-1] != n_frames:
        # F.interpolate wants a channel dimension: (B, T') -> (B, 1, T') -> (B, 1, T) -> (B, T).
        # Nearest neighbour keeps the values exactly 0 / 1 / negative.
        target_with_channel = target.unsqueeze(1)
        resampled = F.interpolate(target_with_channel, size=n_frames, mode='nearest')
        target = resampled.squeeze(1)

    is_known = target >= 0  # (B, T)
    if not is_known.any():
        return frame_logits.sum() * 0

    known_logits = frame_logits[is_known]
    known_targets = target[is_known]
    return F.binary_cross_entropy_with_logits(known_logits, known_targets)
