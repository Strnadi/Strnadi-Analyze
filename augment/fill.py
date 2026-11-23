"""Bring a clip to exactly n samples; a short clip is padded with low-level noise matched to its own tail.
Never zeros (digital silence is a visible artefact the model can latch onto: one BlBh clip was predicted BC
with the Grad-CAM hot spot inside a zero-padded tail) and never wrap-around (`np.resize` copied the clip start
onto its end). Origin: research session, visual-QA fix."""
import numpy as np


def fill_to(y, n, rel=0.3):
    if len(y) >= n:
        return y[:n]
    tail = y[-min(len(y), 3200):] if len(y) else np.zeros(1, np.float32)
    sd = float(np.std(tail)) * rel + 1e-5
    return np.concatenate([y, np.random.normal(0, sd, n - len(y)).astype(np.float32)])
