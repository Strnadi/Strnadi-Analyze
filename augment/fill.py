"""Bring a clip to exactly n samples; a short clip is padded with low-level noise matched to its
own tail.

Never zeros: digital silence is a visible artefact the model can latch onto (one BlBh clip was
predicted BC with the Grad-CAM hot spot inside a zero-padded tail).

Never wrap-around: `np.resize` copied the clip start onto its end.

Origin: research session, visual-QA fix.
"""

import numpy as np

# How much of the clip's end is used to measure its background level: 3200 samples = 0.1 s at
# 32 kHz.
_TAIL_SAMPLES = 3200


def fill_to(y, n, rel=0.3):
    """Cut or pad a waveform to exactly `n` samples.

    A clip that is long enough is cut at the end. A clip that is too short gets Gaussian noise
    appended, whose level follows the last 0.1 s of the clip itself. The noise comes from the
    global `np.random` state.

    Args:
        y: waveform, 1-D float32 array.
        n: wanted length in samples.
        rel: level of the padding noise relative to the standard deviation of the clip's tail
            (0.3 = about 10 dB quieter).

    Returns:
        A waveform of exactly `n` samples. When the clip was long enough this is a view of the
        input, not a copy.
    """
    if len(y) >= n:
        return y[:n]

    # Measure the level at the end of the clip, so the padding continues at a similar loudness.
    if len(y) > 0:
        tail_length = min(len(y), _TAIL_SAMPLES)
        tail = y[-tail_length:]
    else:
        # An empty clip has no tail; a single zero gives a standard deviation of 0.
        tail = np.zeros(1, np.float32)

    # The small constant keeps the padding from being exact digital silence when the tail is
    # silent.
    noise_std = float(np.std(tail)) * rel + 1e-5

    missing_samples = n - len(y)
    padding = np.random.normal(0, noise_std, missing_samples).astype(np.float32)
    return np.concatenate([y, padding])
