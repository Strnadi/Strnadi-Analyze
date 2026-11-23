"""NT-Xent (SimCLR) contrastive loss between two batches of embeddings of the same items under two augmentations.
Embeddings are L2-normalised here. Origin: legacy `model/densenet_simclr.py::nt_xent_loss`."""
import torch
import torch.nn.functional as F


def nt_xent(z_i, z_j, temperature=0.1):
    z = F.normalize(torch.cat([z_i, z_j], 0), dim=1)
    n = len(z_i)
    sim = (z @ z.t()) / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float('-inf'))
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)
