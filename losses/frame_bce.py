"""Frame-level song-activity loss: binary cross-entropy between per-frame logits (B, T) and 0/1 targets given at
any frame rate (resampled to T). Target frames < 0 are ignored (clips without a known song span).
Used to train `model/song_detector.py` and the `frame_logits` of every sequence model. Returns a scalar."""
import torch
import torch.nn.functional as F


def frame_bce(frame_logits, target):
    if target.shape[-1] != frame_logits.shape[-1]:
        target = F.interpolate(target.unsqueeze(1), size=frame_logits.shape[-1], mode='nearest').squeeze(1)
    known = target >= 0
    if not known.any():
        return frame_logits.sum() * 0
    return F.binary_cross_entropy_with_logits(frame_logits[known], target[known])
