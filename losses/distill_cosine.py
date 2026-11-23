"""Embedding distillation: 1 - cos(projected student embedding, teacher embedding), masked for
rows without a teacher target.

Distilling EMBEDDINGS was more reliable than distilling logits in the BirdCLEF 2026 reports, and
the BirdCLEF 2026 winner trained small CNNs exactly this way against Perch 2.0.

Two ways to get teacher targets:
- cached embeddings of the UNAUGMENTED clip (`teacher='cache'`): cheap, but a fixed teacher
  overfits earlier;
- the teacher run on the SAME augmented view as the student (`teacher='online'`, needs a torch
  teacher): "function matching" (Beyer et al. 2022), which with long schedules is what makes
  distillation work.

Returns a scalar.

Origin: research session (implemented, never queued) + both reviews.
"""

import torch.nn.functional as F


def distill_cosine(student_proj, teacher_emb, mask=None):
    """Mean cosine distance between student and teacher embeddings.

    Args:
        student_proj: (B, D) student embeddings, already projected to the teacher's dimension D.
        teacher_emb: (B, D) teacher embeddings.
        mask: optional (B,) tensor, 1 for rows that have a teacher target and 0 for rows that do
            not. None means every row counts.

    Returns:
        Scalar: 1 - cosine similarity, averaged over the rows that count (0 when the mask is all
        zeros).
    """
    cosine_distance = 1 - F.cosine_similarity(student_proj, teacher_emb, dim=1)  # (B,)

    if mask is None:
        return cosine_distance.mean()

    # Average over the masked-in rows only; the clamp avoids 0 / 0 when no row has a target.
    n_rows_with_target = mask.sum().clamp_min(1)
    return (cosine_distance * mask).sum() / n_rows_with_target
