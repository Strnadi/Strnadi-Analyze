"""Frame-level Yellowhammer song detector (sound event detection): a ~0.1 M-parameter CRNN-free conv net that
outputs song activity every 20 ms, plus `detect()` to turn activity into song segments.

Why it is needed (both reviews):
- end to end, missed detections are the largest loss (song recall 0.875 with the energy segmenter);
- the training pools are filtered by BirdNET >= 0.3, which keeps only 74-82 % of songs and biases every pool
  toward loud, typical songs;
- the phone app cannot ship BirdNET; it needs its own detector for live feedback.
BioDCASE 2025 Task 3 showed Yellowhammer detection is easy for tiny models (AP 0.99 at 7.8 kB).
Train with `aux='frames'` (losses/frame_bce.py) on tables that carry `song_t0` / `song_t1`
(Vizina annotations, segmenter + BirdNET pseudo-labels, None windows, hard negatives).
Status: proposed. Untrained.
"""
import torch
import torch.nn as nn
from layers.registry import build_frontend
from layers.coord_freq import CoordFreq
from layers.conv_bn_act import ConvBNAct
from layers.inverted_res_block import InvertedResBlock
from layers.freq_to_channels import FreqToChannels
from layers.temporal_dilated_block import TemporalDilatedBlock
from layers.frame_tag_head import FrameTagHead


class SongDetector(nn.Module):
    frame_hop_s = 0.02

    def __init__(self, n_classes=0, frontend=None, widths=(12, 16, 24, 32), d=64):
        super().__init__()
        fe = dict(n_bins=64); fe.update(frontend or {})
        self.frontend = build_frontend(**fe)
        self.coord = CoordFreq()
        layers = [ConvBNAct(self.frontend.out_channels + 1, widths[0], 3, (2, 2))]; c = widths[0]
        for w in widths[1:]:
            layers.append(InvertedResBlock(c, w, (2, 1))); c = w
        self.cnn = nn.Sequential(*layers)
        self.fold = FreqToChannels()
        self.proj = nn.Sequential(nn.Conv1d(c * (self.frontend.n_bins // 2 ** len(widths)), d, 1, bias=False), nn.BatchNorm1d(d), nn.SiLU())
        self.temporal = nn.Sequential(TemporalDilatedBlock(d, 5, 1), TemporalDilatedBlock(d, 5, 2), TemporalDilatedBlock(d, 5, 4))
        self.tag = FrameTagHead(d, 1)

    def forward(self, x):
        spec = self.frontend(x) if x.dim() == 2 else x
        h = self.temporal(self.proj(self.fold(self.cnn(self.coord(spec)))))
        return {'logits': None, 'frames': h, 'frame_logits': self.tag(h).squeeze(1)}

    @torch.no_grad()
    def detect(self, wave, hi=0.6, lo=0.3, min_dur=0.4, max_gap=0.3):
        """wave: (N,) tensor. Hysteresis thresholding -> list of (t0, t1, mean_probability)."""
        p = torch.sigmoid(self.forward(wave[None])['frame_logits'][0]).cpu().numpy()
        hop, segs, start = self.frame_hop_s, [], None
        for i, v in enumerate(list(p) + [0.0]):
            if v > lo and start is None:
                start = i
            elif v <= lo and start is not None:
                if p[start:i].max() > hi:
                    if segs and start * hop - segs[-1][1] < max_gap:
                        segs[-1] = (segs[-1][0], i * hop, start)
                    else:
                        segs.append((start * hop, i * hop, start))
                start = None
        return [(t0, t1, float(p[int(round(t0 / hop)):int(round(t1 / hop))].mean())) for t0, t1, _ in segs if t1 - t0 >= min_dur]
