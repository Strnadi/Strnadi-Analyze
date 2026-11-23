"""Frame series -> clip prediction: 1x1 projection -> temporal module -> attentive statistics pooling ->
embedding -> class head (+ per-frame activity logits).

Shared by every backbone that produces a (B, C, T) series (timm CNNs, patch transformer, contour TCN, Perch
spatial embeddings), so comparisons between them isolate the backbone. Returns the standard model dict.
Origin: research session (`TemporalHead`).
"""
import torch.nn as nn
from .attentive_stats_pool import AttentiveStatsPool
from .registry import build_temporal, build_head


class SequenceHead(nn.Module):
    def __init__(self, cin, n_classes, d=160, temporal='conv', n_temporal=3, emb_dim=128, drop=0.2, head='linear', head_kw=None):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv1d(cin, d, 1, bias=False), nn.BatchNorm1d(d), nn.SiLU())
        self.temporal = build_temporal(temporal, d, n_temporal, drop / 2)
        self.pool = AttentiveStatsPool(d)
        self.frame_head = nn.Conv1d(d, 1, 1)
        self.emb = nn.Sequential(nn.Linear(2 * d, emb_dim), nn.BatchNorm1d(emb_dim))
        self.drop = nn.Dropout(drop)
        self.cls = build_head(head, emb_dim, n_classes, **(head_kw or {}))

    def forward(self, series):  # (B, C, T)
        h = self.temporal(self.proj(series))
        pooled, att = self.pool(h)
        e = self.emb(pooled)
        out = self.cls(self.drop(e))
        out.update(emb=e, att=att, frames=h, frame_logits=self.frame_head(h).squeeze(1))
        return out
