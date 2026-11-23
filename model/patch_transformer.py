"""Small patch transformer (AST-tiny-like, 3.1 M params) trained from scratch on the log-mel.

Kept as a comparison point only: transformers are data-hungry and there are 3,744 labelled clips; both reviews
expect it to lose to CNNs here. A pretrained small audio transformer would be the fair test.
Origin: research session (`PatchTF`).
"""
import torch
import torch.nn as nn
from layers.registry import build_frontend
from layers.sequence_head import SequenceHead


class PatchTransformer(nn.Module):
    def __init__(self, n_classes=7, frontend=None, patch=(16, 8), dim=192, depth=6, heads=4, drop=0.1, head='linear', head_kw=None):
        super().__init__()
        self.frontend = build_frontend(**(frontend or {}))
        self.embed = nn.Conv2d(self.frontend.out_channels, dim, patch, patch)
        nf = self.frontend.n_bins // patch[0]
        self.fpos = nn.Parameter(torch.zeros(1, dim, nf, 1)); self.tpos = nn.Parameter(torch.zeros(1, dim, 1, 512))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=drop, batch_first=True, norm_first=True, activation='gelu')
        self.tf = nn.TransformerEncoder(layer, depth); self.norm = nn.LayerNorm(dim)
        self.head = SequenceHead(dim * nf, n_classes, 160, 'none', drop=0.2, head=head, head_kw=head_kw)

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        z = self.embed(spec); B, D, Fq, T = z.shape
        z = z + self.fpos + self.tpos[..., :T]
        z = self.norm(self.tf(z.flatten(2).transpose(1, 2)))
        return self.head(z.transpose(1, 2).reshape(B, D * Fq, T))
