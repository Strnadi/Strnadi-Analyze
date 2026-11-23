"""Small speed change (pitch and tempo together), at most +-`max_rel` (default 2 %).

Capped because +-2 % already moves a 1500 Hz offset by 30 Hz and the criteria hinge on 100-250 Hz.

Correction over the research code: that resampled with linear interpolation (`np.interp`), which
low-passes the signal by several dB at 8-11 kHz and adds images; this uses a polyphase resampler.
"""

from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly
from .fill import fill_to


def speed_perturb(y, max_rel=0.02):
    """Play a clip slightly faster or slower, keeping its length.

    The clip is resampled and then treated as if it still had the original sample rate, so pitch
    and tempo change together. A faster clip becomes shorter and is padded back with noise by
    `fill_to`; a slower clip becomes longer and is cut at the end. Uses the global `np.random`
    state.

    Args:
        y: waveform, 1-D float32 array.
        max_rel: largest relative speed change; 0.02 means the speed factor is drawn from
            0.98 ... 1.02.

    Returns:
        A float32 waveform with the same number of samples as `y` (the input itself when the
        drawn factor is so close to 1 that no resampling is needed).
    """
    # A factor above 1 means faster and higher.
    speed_factor = 1 + np.random.uniform(-max_rel, max_rel)

    # The polyphase resampler needs an integer ratio up / down. Playing faster means FEWER
    # samples, so the length ratio is 1 / speed_factor, approximated with a denominator of at
    # most 200.
    length_ratio = Fraction(1 / speed_factor).limit_denominator(200)
    upsample = length_ratio.numerator
    downsample = length_ratio.denominator
    if upsample == downsample:
        # The ratio rounded to exactly 1: nothing to do.
        return y

    resampled = resample_poly(y, upsample, downsample).astype(np.float32)
    return fill_to(resampled, len(y))
