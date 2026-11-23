"""Detect, then zoom: a coarse network reads the whole window, a second network reads a high-resolution crop
around the end of the song, where the dialect and the complete/unfinished decision live.

1. Coarse branch: a small StrnadNet on the standard front-end -> clip embedding.
2. Song end: from the coarse spectrogram, the last frame whose tonal energy (mean of the 3 strongest bins,
   smoothed) exceeds a fraction of the clip maximum. Fixed rule, identical in training and inference.
3. Zoom branch: the waveform from 1.0 s before to 0.5 s after that point, analysed at 64 ms window / 5 ms hop
   on a linear 2.5-9.5 kHz axis (~27 Hz per bin: the 100-250 Hz dialect criteria become 4-9 bins instead of
   1-3), through a long-kernel stem and three residual stages.
4. Both embeddings are concatenated into a factorised head (song / complete / dialect).

Why: given a complete song the current model picks the right dialect 97 % of the time; the errors are missing a
faint second terminal element. The crop is small, so the extra resolution costs little.
Status: proposed (research idea "two-resolution zoom model", never built; fork review idea 3). Untrained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.registry import build_frontend, build_head
from layers.coord_freq import CoordFreq
from layers.tonal_line_conv import TonalLineConv
from layers.res_block import ResBlock
from layers.freq_to_channels import FreqToChannels
from layers.attentive_stats_pool import AttentiveStatsPool
from .strnadnet import StrnadNet

SR = 32000


class EndingZoomNet(nn.Module):
    wave_input = True

    def __init__(self, n_classes=7, coarse_widths=(16, 32, 64, 96), zoom_widths=(24, 48, 96), d=128, emb_dim=128,
                 before_s=1.0, after_s=0.5, rel_threshold=0.3, drop=0.2, head='factorized', head_kw=None):
        super().__init__()
        self.coarse = StrnadNet(n_classes=n_classes, widths=coarse_widths, d=d, emb_dim=emb_dim, drop=drop)
        self.frontend = self.coarse.frontend
        self.before, self.after, self.rel = int(before_s * SR), int(after_s * SR), rel_threshold
        self.zoom_frontend = build_frontend(bank='linear', norm='percentile', n_bins=256, fmin=2500.0, fmax=9500.0, n_fft=2048, hop=160)
        self.coord = CoordFreq()
        layers = [TonalLineConv(2, zoom_widths[0], stride=(2, 2))]; c = zoom_widths[0]
        for w in zoom_widths:
            layers.append(ResBlock(c, w, (2, 2) if w != zoom_widths[-1] else (2, 1))); c = w
        self.zoom_cnn = nn.Sequential(*layers)
        self.fold = FreqToChannels()
        self.zoom_proj = nn.Sequential(nn.Conv1d(c * (256 // 2 ** (len(zoom_widths) + 1)), d, 1, bias=False), nn.BatchNorm1d(d), nn.SiLU())
        self.zoom_pool = AttentiveStatsPool(d)
        self.fuse = nn.Sequential(nn.Linear(2 * d + emb_dim, emb_dim), nn.BatchNorm1d(emb_dim))
        self.drop = nn.Dropout(drop)
        self.cls = build_head(head, emb_dim, n_classes, **(head_kw or {}))

    @torch.no_grad()
    def song_end(self, spec):
        """index (in samples) of the end of the last tonal event in each clip."""
        env = torch.topk(spec[:, 0], 3, dim=1).values.mean(1)                       # (B, T)
        env = F.avg_pool1d(env.unsqueeze(1), 5, 1, 2).squeeze(1)
        active = env > self.rel * env.amax(1, keepdim=True).clamp_min(1e-3)
        last = (active * torch.arange(1, env.shape[1] + 1, device=env.device)).amax(1)
        return last * self.frontend.hop

    def forward(self, wave):
        coarse = self.coarse(wave)
        end = self.song_end(self.frontend(wave))
        L = self.before + self.after
        start = (end - self.before).clamp(0, wave.shape[1] - L)
        crop = wave.gather(1, start[:, None] + torch.arange(L, device=wave.device)[None])
        z = self.zoom_proj(self.fold(self.zoom_cnn(self.coord(self.zoom_frontend(crop)))))
        pooled, zatt = self.zoom_pool(z)
        e = self.fuse(torch.cat([pooled, coarse['emb']], 1))
        out = self.cls(self.drop(e))
        out.update(emb=e, att=coarse['att'], frames=coarse['frames'], frame_logits=coarse['frame_logits'],
                   zoom_start=start, zoom_att=zatt)
        return out
