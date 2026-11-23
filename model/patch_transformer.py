"""Small patch transformer (AST-tiny-like, 3.1 M params) trained from scratch on the log-mel.

Kept as a comparison point only: transformers are data-hungry and there are 3,744 labelled clips;
both reviews expect it to lose to CNNs here. A pretrained small audio transformer would be the fair
test.

Origin: research session (`PatchTF`).
"""

import torch
import torch.nn as nn

from layers.registry import build_frontend
from layers.sequence_head import SequenceHead
from model.base import Model


class PatchTransformer(Model):
    """Front-end -> non-overlapping patches -> transformer encoder -> shared sequence head.

    Args:
        n_classes: number of output classes.
        frontend: dict of keyword arguments for `layers.registry.build_frontend`; None gives its
            defaults (128-bin mel, 10 ms hop).
        patch: (frequency bins, time frames) covered by one patch; patches do not overlap.
        dim: size of a patch token.
        depth: number of transformer encoder layers.
        heads: number of attention heads per layer.
        drop: dropout inside the transformer layers (the sequence head uses a fixed 0.2).
        head: kind of classifier head (`layers.registry.build_head`).
        head_kw: dict of extra keyword arguments for the classifier head.

    `forward` returns the dict of `layers.sequence_head.SequenceHead`: `logits`, `emb`, `att`,
    `frames`, `frame_logits`.
    """

    def __init__(
        self,
        n_classes=7,
        frontend=None,
        patch=(16, 8),
        dim=192,
        depth=6,
        heads=4,
        drop=0.1,
        head='linear',
        head_kw=None,
    ):
        super().__init__()

        # --- front-end: waveform -> normalised spectrogram
        frontend_config = frontend or {}
        self.frontend = build_frontend(**frontend_config)

        # --- patch embedding: one strided convolution cuts the spectrogram into patch tokens
        self.embed = nn.Conv2d(self.frontend.out_channels, dim, patch, patch)

        # --- learned position embeddings, one per patch row (frequency) and per patch column (time)
        patch_rows = self.frontend.n_bins // patch[0]
        max_patch_columns = 512
        self.fpos = nn.Parameter(torch.zeros(1, dim, patch_rows, 1))
        self.tpos = nn.Parameter(torch.zeros(1, dim, 1, max_patch_columns))

        # --- transformer encoder over all patch tokens of the clip
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            4 * dim,
            dropout=drop,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.tf = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)

        # --- frame series -> clip embedding -> classifier; the patch rows of one time step are
        #     stacked into channels, and the head gets no temporal module of its own ('none')
        self.head = SequenceHead(
            dim * patch_rows,
            n_classes,
            160,
            'none',
            drop=0.2,
            head=head,
            head_kw=head_kw,
        )

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 128, 400)   10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        patches = self.embed(spectrogram)  # (B, 192, 8, 50)   8 patch rows, 80 ms columns
        batch, dim, patch_rows, patch_columns = patches.shape
        patches = patches + self.fpos + self.tpos[..., :patch_columns]  # (B, 192, 8, 50)
        tokens = patches.flatten(2).transpose(1, 2)  # (B, 400, 192)   one token per patch
        tokens = self.tf(tokens)  # (B, 400, 192)
        tokens = self.norm(tokens)  # (B, 400, 192)
        series = tokens.transpose(1, 2)  # (B, 192, 400)
        series = series.reshape(batch, dim * patch_rows, patch_columns)  # (B, 1536, 50)
        outputs = self.head(series)  # dict: logits (B, n_classes), emb (B, 128), frames, ...
        return outputs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(PatchTransformer())
