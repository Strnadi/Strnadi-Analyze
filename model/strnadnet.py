"""StrnadNet: small frequency-aware CNN + temporal module + attentive pooling (0.90 M params by
default).

Design, each choice tied to how dialects are defined:
- front-end at ~60-70 Hz per bin where the song lives; a frequency-coordinate channel;
- 4 residual stages that downsample frequency by 16 in total but time only by 4 (40 ms frames);
- the remaining 8 frequency bins are folded into channels, so absolute pitch survives (Bh vs Bl);
- a temporal module with > 1 s receptive field sees element ORDER (BhBl vs BlBh), which pooled
  embeddings lose;
- attentive statistics pooling gives the clip embedding and a "where is the song" weight per frame.

Results (research session, recording-grouped 5-fold CV, 3,744 strong clips): strong labels only
0.745 macro-F1; with weak pools + `head='factorized'` (R2) 0.774. Dialect accuracy given a complete
song is ~97 %; the remaining error is complete-vs-Unfinished (see docs/REVIEW.md).

Origin: research session (`strnadi_ml/models.py`); layer names kept so its checkpoints load
(`training/checkpoint.py::load_research_state`).
"""

import torch.nn as nn

from layers.attentive_stats_pool import AttentiveStatsPool
from layers.conv_bn_act import ConvBNAct
from layers.coord_freq import CoordFreq
from layers.freq_to_channels import FreqToChannels
from layers.inverted_res_block import InvertedResBlock
from layers.registry import build_frontend, build_head, build_temporal
from layers.res_block import ResBlock
from layers.tonal_line_conv import TonalLineConv
from model.base import Model


class StrnadNet(Model):
    """Waveform (or spectrogram) -> class logits, clip embedding and per-frame outputs.

    Args:
        n_classes: number of output classes (the project uses 7: BC, BE, BhBl, BlBh, None,
            Unfinished, XB).
        frontend: dict of keyword arguments for `layers.registry.build_frontend` (bank, norm,
            n_bins, fmin, fmax, n_fft, hop, ...). None gives its defaults: 128-bin mel on
            1.5-11 kHz, log-median normalisation, 1024-sample window, 10 ms hop.
        widths: output channels of the residual stages, one entry per stage. The number of stages
            also fixes the total frequency downsampling, 2 ** len(widths).
        depths: number of residual blocks in each stage (same length as `widths`).
        d: channels of the frame series, i.e. the width of the temporal module.
        n_temporal: number of layers of the temporal module.
        emb_dim: size of the clip embedding.
        dw: True replaces the plain residual blocks of every stage but the first with
            inverted-residual (depthwise) blocks.
        drop: dropout on the clip embedding before the classifier; the temporal module uses half
            of it.
        stem_stride: (frequency, time) stride of the stem.
        time_strides: time stride of the first block of each stage (same length as `widths`).
        coord: True appends the frequency-coordinate channel to the spectrogram.
        stem: 'conv' (one 3x3 convolution) or 'tonal' (`TonalLineConv`: 3x3 plus long time
            kernels).
        temporal: kind of temporal module: 'conv' | 'gru' | 'attn' | 'none'
            (`layers.registry.build_temporal`).
        head: kind of classifier head: 'linear' | 'factorized' | 'cosine' | 'arcface' |
            'measurement' | 'mlp' (`layers.registry.build_head`).
        head_kw: dict of extra keyword arguments for the head ('linear' and 'factorized' take
            none).
        element_tokens: 0, or the size of the element vocabulary (`len(losses.element_ctc.TOKENS)`,
            blank included). The model then also returns per-frame token log-probabilities,
            trained with the CTC loss on element sequences (`losses/element_ctc.py`).

    `forward` returns a dict: `logits` (B, n_classes), `emb` (B, emb_dim), `att` (B, T) frame
    weights, `frames` (B, d, T), `frame_logits` (B, T), plus whatever the chosen head adds.
    """

    def __init__(
        self,
        n_classes=7,
        frontend=None,
        widths=(16, 32, 64, 128),
        depths=(1, 1, 2, 1),
        d=160,
        n_temporal=3,
        emb_dim=128,
        dw=False,
        drop=0.2,
        stem_stride=(2, 2),
        time_strides=(1, 2, 1, 1),
        coord=True,
        stem='conv',
        temporal='conv',
        head='linear',
        head_kw=None,
        element_tokens=0,
    ):
        super().__init__()

        # --- front-end: waveform -> normalised spectrogram
        frontend_config = frontend or {}
        self.frontend = build_frontend(**frontend_config)

        # --- frequency-coordinate channel: lets the convolutions tell absolute pitch
        if coord:
            self.coord = CoordFreq()
        else:
            self.coord = nn.Identity()
        input_channels = self.frontend.out_channels + int(coord)

        # --- stem: the first convolution; strides are (frequency, time)
        stem_stride = tuple(stem_stride)
        if stem == 'tonal':
            self.stem = TonalLineConv(input_channels, widths[0], stride=stem_stride)
        else:
            self.stem = ConvBNAct(input_channels, widths[0], 3, stem_stride)

        # --- 2-D convolutional body: one residual stage per entry of `widths`
        # Total frequency downsampling is always 2 ** len(widths): if the stem already strides in
        # frequency, stage 0 does not.
        first_stage_halving_freq = 1 if stem_stride[0] == 2 else 0
        blocks = []
        channels = widths[0]
        for stage, (stage_width, n_blocks) in enumerate(zip(widths, depths)):
            for block_index in range(n_blocks):
                # only the first block of a stage downsamples
                if block_index == 0:
                    freq_stride = 2 if stage >= first_stage_halving_freq else 1
                    block_stride = (freq_stride, time_strides[stage])
                else:
                    block_stride = (1, 1)
                if dw and stage > 0:
                    block = InvertedResBlock(channels, stage_width, block_stride)
                else:
                    block = ResBlock(channels, stage_width, block_stride)
                blocks.append(block)
                channels = stage_width
        self.cnn = nn.Sequential(*blocks)

        # --- feature map -> frame series: fold the remaining frequency bins into channels
        self.fold = FreqToChannels()
        freq_bins_left = self.frontend.n_bins // (2 ** len(widths))
        self.proj = nn.Sequential(
            nn.Conv1d(channels * freq_bins_left, d, 1, bias=False),
            nn.BatchNorm1d(d),
            nn.SiLU(),
        )

        # --- temporal module: sees the order of the elements
        self.temporal = build_temporal(temporal, d, n_temporal, drop / 2)

        # --- frame series -> clip embedding
        self.pool = AttentiveStatsPool(d)
        self.frame_head = nn.Conv1d(d, 1, 1)  # per-frame song-activity logit
        # per-frame element tokens (phrase, B, C, ...), decoded as a sequence
        self.token_head = nn.Conv1d(d, element_tokens, 1) if element_tokens else None
        self.emb = nn.Sequential(
            nn.Linear(2 * d, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

        # --- classifier head
        self.drop = nn.Dropout(drop)
        head_config = head_kw or {}
        self.cls = build_head(head, emb_dim, n_classes, **head_config)

    def frames(self, x):
        """Waveform (B, N) or spectrogram (B, C, F, T) -> frame series (B, d, T / 4).

        Shape comments: default arguments, 4 s of audio.
        """
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 128, 400)   10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        cnn_input = self.coord(spectrogram)  # (B, 2, 128, 400)   + frequency coordinate
        stem_map = self.stem(cnn_input)  # (B, 16, 64, 200)
        feature_map = self.cnn(stem_map)  # (B, 128, 8, 100)   40 ms frames
        series = self.fold(feature_map)  # (B, 1024, 100)
        series = self.proj(series)  # (B, 160, 100)
        series = self.temporal(series)  # (B, 160, 100)
        return series

    def forward(self, x):
        frames = self.frames(x)  # (B, 160, 100)
        pooled, frame_weights = self.pool(frames)  # (B, 320), (B, 100)
        embedding = self.emb(pooled)  # (B, 128)
        outputs = self.cls(self.drop(embedding))  # dict with `logits` (B, n_classes)
        frame_logits = self.frame_head(frames).squeeze(1)  # (B, 100)
        outputs.update(emb=embedding, att=frame_weights, frames=frames, frame_logits=frame_logits)
        if self.token_head is not None:
            token_logits = self.token_head(frames).transpose(1, 2)  # (B, 100, n_tokens)
            outputs['token_log_probs'] = token_logits.float().log_softmax(-1)
        return outputs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(StrnadNet())
