"""Frame-level element-tag loss: cross-entropy between (B, n_tags, T) logits and integer tags (B, T'),
resampled to T; tag -1 = unknown, ignored. Used by `model/element_tagger.py`. Returns a scalar."""
import torch.nn.functional as F


def frame_ce(tag_logits, tags):
    if tags.shape[-1] != tag_logits.shape[-1]:
        tags = F.interpolate(tags[:, None].float(), size=tag_logits.shape[-1], mode='nearest').squeeze(1).long()
    if not (tags >= 0).any():
        return tag_logits.sum() * 0
    return F.cross_entropy(tag_logits, tags, ignore_index=-1)
