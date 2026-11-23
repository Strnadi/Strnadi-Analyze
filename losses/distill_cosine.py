"""Embedding distillation: 1 - cos(projected student embedding, teacher embedding), masked for rows without a
teacher target. Distilling EMBEDDINGS was more reliable than distilling logits in the BirdCLEF 2026 reports,
and the BirdCLEF 2026 winner trained small CNNs exactly this way against Perch 2.0.

Two ways to get teacher targets:
- cached embeddings of the UNAUGMENTED clip (`teacher='cache'`): cheap, but a fixed teacher overfits earlier;
- the teacher run on the SAME augmented view as the student (`teacher='online'`, needs a torch teacher):
  "function matching" (Beyer et al. 2022), which with long schedules is what makes distillation work.
Returns a scalar. Origin: research session (implemented, never queued) + both reviews.
"""
import torch.nn.functional as F


def distill_cosine(student_proj, teacher_emb, mask=None):
    loss = 1 - F.cosine_similarity(student_proj, teacher_emb, dim=1)
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp_min(1)
