"""Frame-level Yellowhammer song detector (sound event detection): a 0.07 M-parameter conv
net that outputs song activity every 20 ms, plus `detect()` to turn activity into song segments.

Why it is needed (both reviews):
- end to end, missed detections are the largest loss (song recall 0.875 with the energy
  segmenter);
- the training pools are filtered by BirdNET >= 0.3, which keeps only 74-82 % of songs and biases
  every pool toward loud, typical songs;
- the phone app cannot ship BirdNET; it needs its own detector for live feedback.

BioDCASE 2025 Task 3 showed Yellowhammer detection is easy for tiny models (AP 0.99 at 7.8 kB).
Train with `aux='frames'` (losses/frame_bce.py) on tables that carry `song_t0` / `song_t1` (Vizina
annotations, segmenter + BirdNET pseudo-labels, None windows, hard negatives).

Status: proposed. Untrained.
"""

import torch
import torch.nn as nn

from layers.conv_bn_act import ConvBNAct
from layers.coord_freq import CoordFreq
from layers.frame_tag_head import FrameTagHead
from layers.freq_to_channels import FreqToChannels
from layers.inverted_res_block import InvertedResBlock
from layers.registry import build_frontend
from layers.temporal_dilated_block import TemporalDilatedBlock
from model.base import Model


class SongDetector(Model):
    """Waveform (or spectrogram) -> one song-activity logit per 20 ms frame.

    Args:
        n_classes: not used (the detector has no clip classifier); accepted and ignored.
        frontend: dict of front-end settings (see `layers.registry.build_frontend`) that override
            this model's own default of 64 mel bins (everything else as in `build_frontend`).
        widths: output channels of the stem (first entry) and of the inverted-residual stages (the
            rest). The stem and every stage halve frequency, 2 ** len(widths) in total.
        d: channels of the frame series.

    `forward` returns a dict: `logits` = None, `frames` (B, d, T) and `frame_logits` (B, T), with
    T one frame per 20 ms. `detect` turns one waveform into a list of song segments.
    """

    frame_hop_s = 0.02  # seconds per output frame

    def __init__(self, n_classes=0, frontend=None, widths=(12, 16, 24, 32), d=64):
        super().__init__()

        # --- front-end: waveform -> normalised 64-bin spectrogram, plus the frequency coordinate
        frontend_config = dict(n_bins=64)
        frontend_config.update(frontend or {})
        self.frontend = build_frontend(**frontend_config)
        self.coord = CoordFreq()
        input_channels = self.frontend.out_channels + 1

        # --- 2-D convolutional body: stem (halves frequency and time), then inverted-residual
        #     stages that halve frequency only, so frames stay at 20 ms
        layers = [ConvBNAct(input_channels, widths[0], 3, (2, 2))]
        channels = widths[0]
        for stage_width in widths[1:]:
            stage_stride = (2, 1)
            layers.append(InvertedResBlock(channels, stage_width, stage_stride))
            channels = stage_width
        self.cnn = nn.Sequential(*layers)

        # --- feature map -> frame series: fold the remaining frequency bins into channels
        self.fold = FreqToChannels()
        freq_bins_left = self.frontend.n_bins // 2 ** len(widths)
        self.proj = nn.Sequential(
            nn.Conv1d(channels * freq_bins_left, d, 1, bias=False),
            nn.BatchNorm1d(d),
            nn.SiLU(),
        )

        # --- temporal module: three dilated blocks (dilation 1, 2, 4)
        self.temporal = nn.Sequential(
            TemporalDilatedBlock(d, 5, 1),
            TemporalDilatedBlock(d, 5, 2),
            TemporalDilatedBlock(d, 5, 4),
        )

        # --- per-frame song-activity classifier
        self.tag = FrameTagHead(d, 1)

    def forward(self, x):
        # Shape comments: default arguments, 4 s of audio.
        if x.dim() == 2:
            spectrogram = self.frontend(x)  # (B, 1, 64, 400)   10 ms frames
        else:
            spectrogram = x  # already a spectrogram from `self.frontend`
        cnn_input = self.coord(spectrogram)  # (B, 2, 64, 400)   + frequency coordinate
        feature_map = self.cnn(cnn_input)  # (B, 32, 4, 200)   20 ms frames
        series = self.fold(feature_map)  # (B, 128, 200)
        series = self.proj(series)  # (B, 64, 200)
        frames = self.temporal(series)  # (B, 64, 200)
        frame_logits = self.tag(frames).squeeze(1)  # (B, 1, 200) -> (B, 200)
        return {'logits': None, 'frames': frames, 'frame_logits': frame_logits}

    @torch.no_grad()
    def detect(self, wave, hi=0.6, lo=0.3, min_dur=0.4, max_gap=0.3):
        """Find the songs in one waveform by hysteresis thresholding of the frame probabilities.

        Args:
            wave: (N,) waveform tensor.
            hi: a segment is kept only if its probability exceeds this somewhere.
            lo: a segment lasts for as long as the probability stays above this.
            min_dur: segments shorter than this many seconds are dropped.
            max_gap: a segment starting less than this many seconds after the previous one is
                merged with it.

        Returns a list of (t0, t1, mean_probability), times in seconds.
        """
        frame_logits = self.forward(wave[None])['frame_logits'][0]  # (T,)
        probability = torch.sigmoid(frame_logits).cpu().numpy()  # (T,)
        hop = self.frame_hop_s

        # --- walk through the frames and collect (start time, end time) of confident segments
        segments = []
        start = None  # first frame of the segment we are inside, or None between segments
        frames_then_end = list(probability) + [0.0]  # the extra 0 closes a segment at the end
        for frame, value in enumerate(frames_then_end):
            if value > lo and start is None:
                start = frame
            elif value <= lo and start is not None:
                if probability[start:frame].max() > hi:
                    follows_closely = len(segments) > 0 and start * hop - segments[-1][1] < max_gap
                    if follows_closely:
                        segments[-1] = (segments[-1][0], frame * hop)  # extend the previous one
                    else:
                        segments.append((start * hop, frame * hop))
                start = None

        # --- drop short segments and attach the mean probability of each
        detections = []
        for t0, t1 in segments:
            if t1 - t0 >= min_dur:
                first_frame = int(round(t0 / hop))
                end_frame = int(round(t1 / hop))
                mean_probability = float(probability[first_frame:end_frame].mean())
                detections.append((t0, t1, mean_probability))
        return detections


# %% Quick look: build the model and show what it returns for 4 s of audio
if __name__ == '__main__':
    from model.base import describe

    describe(SongDetector())
