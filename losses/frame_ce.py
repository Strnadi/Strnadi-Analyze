"""Frame-level element-tag loss: cross-entropy between (B, n_tags, T) logits and integer tags
(B, T'), resampled to T; tag -1 = unknown, ignored.

Used by `model/element_tagger.py`.

Returns a scalar.
"""

import torch.nn.functional as F


def frame_ce(tag_logits, tags):
    """Cross-entropy between per-frame element-tag logits and integer tags.

    Args:
        tag_logits: (B, n_tags, T) logits, one vector of tag scores per time frame.
        tags: (B, T') integer tag per frame, -1 = unknown (ignored). T' may differ from T; the
            tags are then resampled to T by nearest neighbour.

    Returns:
        Scalar: mean loss over the frames with a known tag. Zero (still connected to
        `tag_logits` for autograd) when no frame has one.
    """
    n_frames = tag_logits.shape[-1]

    if tags.shape[-1] != n_frames:
        # F.interpolate needs floats and a channel dimension:
        # (B, T') -> (B, 1, T') -> (B, 1, T) -> (B, T), then back to integers.
        tags_with_channel = tags[:, None].float()
        resampled = F.interpolate(tags_with_channel, size=n_frames, mode='nearest')
        tags = resampled.squeeze(1).long()

    is_known = tags >= 0  # (B, T)
    if not is_known.any():
        return tag_logits.sum() * 0

    return F.cross_entropy(tag_logits, tags, ignore_index=-1)
