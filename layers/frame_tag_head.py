"""Per-frame classifier over the frame series: (B, D, T) -> (B, n_tags, T) logits.

Used for (a) song-activity detection (n_tags = 1, sigmoid) and (b) element tagging (softmax over
background / phrase / B / C / D / E / X), the TweetyNet / DAS formulation of birdsong annotation.
Origin: research StrnadNet had an unused `frame_head`; both reviews ask for a frame-level detector, the fork
review for element-sequence output.
"""
import torch.nn as nn


class FrameTagHead(nn.Module):
    def __init__(self, d, n_tags=1):
        super().__init__()
        self.conv = nn.Conv1d(d, n_tags, 1)

    def forward(self, frames):
        return self.conv(frames)
