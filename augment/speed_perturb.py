"""Small speed change (pitch and tempo together), at most +-`max_rel` (default 2 %).

Capped because +-2 % already moves a 1500 Hz offset by 30 Hz and the criteria hinge on 100-250 Hz.
Correction over the research code: that resampled with linear interpolation (`np.interp`), which low-passes
the signal by several dB at 8-11 kHz and adds images; this uses a polyphase resampler.
"""
from fractions import Fraction
import numpy as np
from scipy.signal import resample_poly
from .fill import fill_to


def speed_perturb(y, max_rel=0.02):
    f = 1 + np.random.uniform(-max_rel, max_rel)              # > 1: faster and higher
    fr = Fraction(1 / f).limit_denominator(200)
    if fr.numerator == fr.denominator:
        return y
    out = resample_poly(y, fr.numerator, fr.denominator).astype(np.float32)
    return fill_to(out, len(y))
