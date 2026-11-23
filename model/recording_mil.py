"""Bird-level model: all songs of one recording -> one dialect decision with an attention weight per
song.

Input is a bag of song embeddings (B, N, D) from a frozen song model, with a mask for padding; a
small transformer lets songs inform each other (a male repeats his song), gated-attention pooling
picks the songs that carry the dialect (the complete ones), and the head outputs the 5 dialects +
"no complete song".

Why: the scientific output is a dialect per bird, not per clip, and under this view the 3,333
file-level dialect labels are CORRECT bag supervision ("this recording contains at least one
complete song of D") instead of noisy per-clip labels. It also detects mixed singers through the
attention pattern.

Train with `training/bird_level.py` (`experiments/recording_mil.py`) on embeddings cached by `evaluation/embed_songs.py`.

Status: proposed (research ideas "bout-level set transformer" and "attention-MIL", never built;
fork review idea 8). Untrained.
"""

import torch.nn as nn

from layers.attention_mil_pool import AttentionMILPool
from model.base import Model

# output classes: the 5 dialects + "no complete song"
BAG_CLASSES = ['BC', 'BE', 'BhBl', 'BlBh', 'XB', 'none']


class RecordingMIL(Model):
    """Bag of song embeddings of one recording -> one decision for the recording.

    Args:
        d_in: size of one input song embedding.
        d: width of the model (song tokens and bag embedding).
        n_layers: number of transformer encoder layers; 0 skips the transformer, so songs are
            pooled without looking at each other.
        heads: number of attention heads per transformer layer.
        drop: dropout inside the transformer layers.
        n_out: number of output classes.

    `forward(x, mask)` takes `x` (B, N, d_in), N songs per recording (padded), and an optional
    `mask` (B, N), True = real song. It returns a dict: `logits` (B, n_out), `emb` (B, d), the bag
    embedding, and `att` (B, N), the weight of every song (the weights of a recording sum to 1).
    """

    def __init__(self, d_in=128, d=128, n_layers=1, heads=4, drop=0.1, n_out=len(BAG_CLASSES)):
        super().__init__()

        # --- input projection: song embedding -> song token
        self.inp = nn.Sequential(
            nn.Linear(d_in, d),
            nn.LayerNorm(d),
        )

        # --- transformer encoder: lets the songs of one recording inform each other
        layer = nn.TransformerEncoderLayer(
            d,
            heads,
            2 * d,
            dropout=drop,
            batch_first=True,
            norm_first=True,
        )
        if n_layers:
            self.enc = nn.TransformerEncoder(layer, n_layers)
        else:
            self.enc = None

        # --- song tokens -> bag embedding: gated attention picks the songs that matter
        self.pool = AttentionMILPool(d)

        # --- classifier head
        self.cls = nn.Linear(d, n_out)

    def forward(self, x, mask=None):  # x (B, N, D), mask (B, N) True = real song
        # Shape comments: default arguments.
        tokens = self.inp(x)  # (B, N, 128)
        if self.enc is not None:
            if mask is None:
                padding_mask = None
            else:
                padding_mask = ~mask.bool()  # (B, N)   True = padding, the transformer ignores it
            tokens = self.enc(tokens, src_key_padding_mask=padding_mask)  # (B, N, 128)
        bag_embedding, song_weights = self.pool(tokens, mask)  # (B, 128), (B, N)
        logits = self.cls(bag_embedding)  # (B, 6)
        return {'logits': logits, 'emb': bag_embedding, 'att': song_weights}


# %% Quick look: build the model and show what it returns for 3 recordings of 9 songs each
if __name__ == '__main__':
    import torch

    from model.base import count_parameters

    model = RecordingMIL().eval()
    song_embeddings = torch.randn(3, 9, 128)
    with torch.no_grad():
        outputs = model(song_embeddings)
    print(f'RecordingMIL: {count_parameters(model) / 1e6:.2f} M parameters')
    print(f'  input  {tuple(song_embeddings.shape)}')
    for name, value in outputs.items():
        print(f'  {name:16s} {tuple(value.shape)}')
