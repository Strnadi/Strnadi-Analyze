"""Bird-level model: all songs of one recording -> one dialect decision with an attention weight per song.

Input is a bag of song embeddings (B, N, D) from a frozen song model, with a mask for padding; a small
transformer lets songs inform each other (a male repeats his song), gated-attention pooling picks the songs
that carry the dialect (the complete ones), and the head outputs the 5 dialects + "no complete song".

Why: the scientific output is a dialect per bird, not per clip, and under this view the 3,333 file-level
dialect labels are CORRECT bag supervision ("this recording contains at least one complete song of D") instead
of noisy per-clip labels. It also detects mixed singers through the attention pattern.
Train with `python -m training.train_mil` on embeddings cached by `evaluation/embed_songs.py`.
Status: proposed (research ideas "bout-level set transformer" and "attention-MIL", never built; fork review
idea 8). Untrained.
"""
import torch.nn as nn
from layers.attention_mil_pool import AttentionMILPool

BAG_CLASSES = ['BC', 'BE', 'BhBl', 'BlBh', 'XB', 'none']


class RecordingMIL(nn.Module):
    def __init__(self, d_in=128, d=128, n_layers=1, heads=4, drop=0.1, n_out=len(BAG_CLASSES)):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers) if n_layers else None
        self.pool = AttentionMILPool(d)
        self.cls = nn.Linear(d, n_out)

    def forward(self, x, mask=None):                                    # (B, N, D), mask (B, N) True = real song
        h = self.inp(x)
        if self.enc is not None:
            h = self.enc(h, src_key_padding_mask=None if mask is None else ~mask.bool())
        bag, att = self.pool(h, mask)
        return {'logits': self.cls(bag), 'emb': bag, 'att': att}
