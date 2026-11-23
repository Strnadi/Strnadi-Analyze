"""Perch 2.0 spatial embedding kept as a time series -> shared sequence head.

Three uses, in increasing cost:
1. `precomputed=True`: input is the cached spatial embedding (B, T', 1536) (frequency already
   averaged). Frozen Perch + this head scores 0.741 macro-F1, i.e. the from-scratch StrnadNet
   baseline (0.745), and a fold trains in minutes. This is the fast track for searching weak-label
   recipes (factorised head, bag loss, pool shares) before spending fold-hours on them (fork
   review, step 6).
2. `precomputed=False, trainable=False`: the torch port of Perch runs inside the model (frozen).
3. `trainable=True`: Perch fine-tuned end to end = the strong teacher both reviews ask for. Use
   `backbone_lr_factor` ~0.1 and a small base lr (1e-4 .. 3e-4).

Origin: research session (frozen probe, `eval_frozen_emb.py`) + both reviews (fine-tuning, never
started there).
"""

from layers.sequence_head import SequenceHead
from model.base import Model


class PerchTemporal(Model):
    """Perch embedding per time step -> shared sequence head.

    Args:
        n_classes: number of output classes.
        precomputed: True (default) = the input is the cached spatial embedding (B, T', 1536) and
            no Perch network is built; False = the input is a waveform and the torch port of Perch
            runs inside the model (downloaded on first use).
        trainable: only with `precomputed=False`: True fine-tunes Perch, False keeps it frozen.
        d: channels of the frame series inside the sequence head.
        temporal: kind of temporal module in the head: 'conv' | 'gru' | 'attn' | 'none'.
        n_temporal: number of layers of the temporal module.
        drop: dropout in the sequence head.
        head: kind of classifier head (`layers.registry.build_head`).
        head_kw: dict of extra keyword arguments for the classifier head.

    `forward` returns the dict of `layers.sequence_head.SequenceHead`: `logits`, `emb`, `att`,
    `frames`, `frame_logits`.
    """

    wave_input = True  # the trainer passes the input as is (no spectrogram augmentation)
    backbone_prefixes = ('encoder.',)  # parameter-name prefixes that get `backbone_lr_factor`

    def __init__(
        self,
        n_classes=7,
        precomputed=True,
        trainable=False,
        d=192,
        temporal='conv',
        n_temporal=2,
        drop=0.2,
        head='linear',
        head_kw=None,
    ):
        super().__init__()

        # --- no spectrogram front-end of our own: Perch brings its own (or the input is cached)
        self.frontend = None

        # --- encoder: Perch 2.0, only when the embeddings are not precomputed
        if precomputed:
            self.encoder = None
        else:
            from layers.perch_encoder import PerchEncoder  # needs a network download

            self.encoder = PerchEncoder(trainable=trainable)

        # --- frame series -> clip embedding -> classifier (shared with the other backbones)
        self.head = SequenceHead(
            1536,
            n_classes,
            d,
            temporal,
            n_temporal,
            drop=drop,
            head=head,
            head_kw=head_kw,
        )

    def forward(self, x):
        if self.encoder is None:
            # precomputed spatial embedding: (B, T', 1536) -> (B, 1536, T')
            series = x.transpose(1, 2)
        else:
            # waveform (B, N) -> Perch feature map (B, 1536, T', F') -> average over frequency
            feature_map = self.encoder(x)['spatial']  # (B, 1536, T', F')
            series = feature_map.mean(-1)  # (B, 1536, T')
        outputs = self.head(series)  # dict: logits (B, n_classes), emb (B, 128), frames, ...
        return outputs


# %% Quick look: build the model and show what it returns for cached Perch spatial embeddings
if __name__ == '__main__':
    import torch

    from model.base import describe

    describe(PerchTemporal(), example=torch.randn(2, 16, 1536))
