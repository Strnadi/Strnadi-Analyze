"""Detect, then zoom: a coarse network reads the whole window, a second network reads a
high-resolution crop around the end of the song, where the dialect and the complete/unfinished
decision live.

1. Coarse branch: a small StrnadNet on the standard front-end -> clip embedding.
2. Song end: from the coarse spectrogram, the last frame whose tonal energy (mean of the 3 strongest
   bins, smoothed) exceeds a fraction of the clip maximum. Fixed rule, identical in training and
   inference.
3. Zoom branch: the waveform from 0.6 s before to 1.2 s after that point, analysed at 64 ms window /
   5 ms hop on a linear 2.5-9.5 kHz axis (~27 Hz per bin: the 100-250 Hz dialect criteria become
   4-9 bins instead of 1-3), through a long-kernel stem and three residual stages.
4. Both embeddings are concatenated into a factorised head (song / complete / dialect).

Why: given a complete song the current model picks the right dialect 97 % of the time; the errors
are missing a faint second terminal element. The crop is small, so the extra resolution costs
little.

Status: proposed (research idea "two-resolution zoom model", never built; fork review idea 3).
Untrained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attentive_stats_pool import AttentiveStatsPool
from layers.coord_freq import CoordFreq
from layers.freq_to_channels import FreqToChannels
from layers.registry import build_frontend, build_head
from layers.res_block import ResBlock
from layers.tonal_line_conv import TonalLineConv
from model.base import Model

from .strnadnet import StrnadNet

SR = 32000  # sample rate in Hz


class EndingZoomNet(Model):
    """Whole-clip StrnadNet + a high-resolution network on a crop around the end of the song.

    Args:
        n_classes: number of output classes.
        coarse_widths: stage widths of the coarse branch (the `widths` of its `StrnadNet`).
        zoom_widths: output channels of the residual stages of the zoom branch, one per stage.
        d: channels of the frame series, in both branches.
        emb_dim: size of the coarse clip embedding and of the fused embedding.
        before_s: seconds of audio the crop keeps before the detected song end.
        after_s: seconds of audio the crop keeps after the detected song end. The rule in step 2
            often stops at the B element; the quiet terminal C / E element of BC / BE songs follows
            up to 1.1 s later (research log 17.4), so the crop has to reach that far.
        rel_threshold: a frame counts as song when its tonal energy exceeds this fraction of the
            clip maximum (used to find the song end).
        drop: dropout on the fused embedding before the classifier (also passed to the coarse
            branch).
        head: kind of classifier head (`layers.registry.build_head`).
        head_kw: dict of extra keyword arguments for the classifier head.
        locate: where the crop is centred. 'energy' = the end of the last tonal event (the fixed
            rule of step 2). 'attention' = the frame where the coarse branch's own temporal
            attention peaks, which in trained models is the song's last element; the model then
            looks twice at the place it already finds most informative, and the faint C / E
            element that the energy rule misses is inside the crop by construction.

    The input must be a waveform (B, N) of at least `before_s + after_s` seconds. `forward`
    returns a dict: `logits` (B, n_classes) and `emb` (B, emb_dim) from the fused embedding (plus
    whatever the chosen head adds); `att`, `frames` and `frame_logits` from the coarse branch;
    `zoom_start` (B,), the first sample of each crop; `zoom_att` (B, T_zoom), the frame weights of
    the zoom branch.
    """

    wave_input = True  # the trainer passes the input as is (no spectrogram augmentation)

    def __init__(
        self,
        n_classes=7,
        coarse_widths=(16, 32, 64, 96),
        zoom_widths=(24, 48, 96),
        d=128,
        emb_dim=128,
        before_s=0.6,
        after_s=1.2,
        rel_threshold=0.3,
        drop=0.2,
        head='factorized',
        head_kw=None,
        locate='energy',
    ):
        super().__init__()
        self.locate = locate

        # --- coarse branch: a small StrnadNet that reads the whole clip
        self.coarse = StrnadNet(
            n_classes=n_classes,
            widths=coarse_widths,
            d=d,
            emb_dim=emb_dim,
            drop=drop,
        )
        self.frontend = self.coarse.frontend  # the same module, also used to find the song end

        # --- crop around the song end: lengths in samples, and the song-activity threshold
        self.before = int(before_s * SR)
        self.after = int(after_s * SR)
        self.rel = rel_threshold

        # --- zoom front-end: 64 ms window, 5 ms hop, 256 linear bins on 2.5-9.5 kHz
        self.zoom_frontend = build_frontend(
            bank='linear',
            norm='percentile',
            n_bins=256,
            fmin=2500.0,
            fmax=9500.0,
            n_fft=2048,
            hop=160,
        )
        self.coord = CoordFreq()  # adds the frequency-coordinate channel: 1 -> 2 input channels

        # --- zoom 2-D body: long-kernel stem, then one residual stage per entry of `zoom_widths`
        layers = [TonalLineConv(2, zoom_widths[0], stride=(2, 2))]
        channels = zoom_widths[0]
        for stage_width in zoom_widths:
            # Every stage halves frequency and time, except that a stage as wide as the last one
            # keeps its time resolution (the widths are compared, not the stage positions).
            if stage_width != zoom_widths[-1]:
                stage_stride = (2, 2)
            else:
                stage_stride = (2, 1)
            layers.append(ResBlock(channels, stage_width, stage_stride))
            channels = stage_width
        self.zoom_cnn = nn.Sequential(*layers)

        # --- zoom feature map -> frame series: fold the remaining frequency bins into channels
        # (the stem and every stage halve the 256 bins)
        self.fold = FreqToChannels()
        freq_bins_left = 256 // 2 ** (len(zoom_widths) + 1)
        self.zoom_proj = nn.Sequential(
            nn.Conv1d(channels * freq_bins_left, d, 1, bias=False),
            nn.BatchNorm1d(d),
            nn.SiLU(),
        )

        # --- zoom frame series -> zoom embedding
        self.zoom_pool = AttentiveStatsPool(d)

        # --- fuse the zoom embedding (2 * d) with the coarse clip embedding (emb_dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d + emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

        # --- classifier head
        self.drop = nn.Dropout(drop)
        head_config = head_kw or {}
        self.cls = build_head(head, emb_dim, n_classes, **head_config)

    @torch.no_grad()
    def song_end(self, spec):
        """Index (in samples) of the end of the last tonal event in each clip.

        `spec` is the coarse spectrogram (B, C, F, T); returns a (B,) integer tensor (0 where no
        frame passes the threshold).
        """
        first_channel = spec[:, 0]  # (B, F, T)
        strongest_bins = torch.topk(first_channel, 3, dim=1).values  # (B, 3, T)
        envelope = strongest_bins.mean(1)  # (B, T)   tonal energy per frame
        envelope = F.avg_pool1d(envelope.unsqueeze(1), 5, 1, 2).squeeze(1)  # (B, T)   smoothed
        clip_maximum = envelope.amax(1, keepdim=True).clamp_min(1e-3)  # (B, 1)
        active = envelope > self.rel * clip_maximum  # (B, T)   True where there is song
        frame_numbers = torch.arange(1, envelope.shape[1] + 1, device=envelope.device)  # 1 .. T
        last_active_frame = (active * frame_numbers).amax(1)  # (B,)
        return last_active_frame * self.frontend.hop  # frames -> samples

    def forward(self, wave):
        # Shape comments: default arguments, 4 s of audio.

        # --- coarse branch on the whole clip
        coarse = self.coarse(wave)  # dict: emb (B, 128), att (B, 100), frames (B, 128, 100), ...

        # --- where does the song end?
        if self.locate == 'attention':
            samples_per_frame = wave.shape[1] // coarse['att'].shape[1]
            end = coarse['att'].detach().argmax(1) * samples_per_frame  # (B,)   in samples
        else:
            coarse_spectrogram = self.frontend(wave)  # (B, 1, 128, 400)
            end = self.song_end(coarse_spectrogram)  # (B,)   in samples

        # --- cut the crop [end - before, end + after) out of the waveform, kept inside the clip
        crop_length = self.before + self.after  # 57600 samples = 1.8 s
        start = (end - self.before).clamp(0, wave.shape[1] - crop_length)  # (B,)
        offsets = torch.arange(crop_length, device=wave.device)  # (57600,)
        sample_index = start[:, None] + offsets[None]  # (B, 57600)
        crop = wave.gather(1, sample_index)  # (B, 57600)

        # --- zoom branch on the crop
        zoom_spectrogram = self.zoom_frontend(crop)  # (B, 1, 256, 360)   5 ms frames
        zoom_input = self.coord(zoom_spectrogram)  # (B, 2, 256, 360)   + frequency coordinate
        zoom_map = self.zoom_cnn(zoom_input)  # (B, 96, 16, 38)
        zoom_series = self.fold(zoom_map)  # (B, 1536, 38)
        zoom_series = self.zoom_proj(zoom_series)  # (B, 128, 38)
        zoom_pooled, zoom_weights = self.zoom_pool(zoom_series)  # (B, 256), (B, 38)

        # --- fuse both branches and classify
        both = torch.cat([zoom_pooled, coarse['emb']], 1)  # (B, 384)
        embedding = self.fuse(both)  # (B, 128)
        outputs = self.cls(self.drop(embedding))  # dict with `logits` (B, n_classes)
        outputs.update(
            emb=embedding,
            att=coarse['att'],
            frames=coarse['frames'],
            frame_logits=coarse['frame_logits'],
            zoom_start=start,
            zoom_att=zoom_weights,
        )
        return outputs


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(EndingZoomNet())
