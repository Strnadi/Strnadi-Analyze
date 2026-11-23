"""NT-Xent (SimCLR) contrastive loss between two batches of embeddings of the same items under two
augmentations.

Embeddings are L2-normalised here.

Origin: legacy `model/densenet_simclr.py::nt_xent_loss`.
"""

import torch
import torch.nn.functional as F


def nt_xent(z_i, z_j, temperature=0.1):
    """Contrastive loss: each embedding must find the other view of its own item.

    Args:
        z_i: (B, D) embeddings of the first augmented view.
        z_j: (B, D) embeddings of the second augmented view; row k is the same item as row k of
            `z_i`.
        temperature: divides the cosine similarities; smaller = sharper.

    Returns:
        Scalar: cross-entropy averaged over all 2B embeddings.
    """
    batch_size = len(z_i)

    # Stack both views: rows 0 ... B-1 are view 1, rows B ... 2B-1 are view 2.
    embeddings = torch.cat([z_i, z_j], 0)  # (2B, D)
    embeddings = F.normalize(embeddings, dim=1)

    # Cosine similarity of every embedding with every other one, as logits.
    similarity = (embeddings @ embeddings.t()) / temperature  # (2B, 2B)

    # An embedding must not pick itself: remove the diagonal from the choices.
    is_self = torch.eye(2 * batch_size, dtype=torch.bool, device=embeddings.device)
    similarity = similarity.masked_fill(is_self, float('-inf'))

    # The correct choice for row k is the other view of the same item: k + B for the first half,
    # k - B for the second half.
    partner_of_first_view = torch.arange(batch_size, 2 * batch_size)
    partner_of_second_view = torch.arange(0, batch_size)
    partner = torch.cat([partner_of_first_view, partner_of_second_view]).to(embeddings.device)

    return F.cross_entropy(similarity, partner)
