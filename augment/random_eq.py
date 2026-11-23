"""Random smooth spectral gain curve (+-6 dB knots plus a tilt): a cheap stand-in for unknown
microphone / device frequency responses.

Origin: research session (`ssl_pretrain.py`, now `training/pretrain.py`), used for SSL views.
"""

import numpy as np

# Number of points, evenly spaced from 0 Hz to the Nyquist frequency, at which a random gain is
# drawn; the curve between them is linear interpolation.
_N_KNOTS = 6


def random_eq(y):
    """Apply a random, smooth equaliser curve to a clip.

    The gain in dB is the sum of a piecewise-linear curve through 6 random knots (each within
    +-6 dB) and a straight tilt across the whole band (up to +-3 dB at either edge). Uses the
    global `np.random` state.

    Args:
        y: waveform, 1-D float array.

    Returns:
        The equalised float32 waveform, same length as `y`.
    """
    spectrum = np.fft.rfft(y)

    # Position of every FFT bin on a 0 ... 1 scale (0 = 0 Hz, 1 = Nyquist frequency).
    relative_frequency = np.linspace(0, 1, len(spectrum))

    # Smooth part: random gains at the knots, linearly interpolated in between.
    knot_positions = np.linspace(0, 1, _N_KNOTS)
    knot_gains_db = np.random.uniform(-6, 6, _N_KNOTS)
    smooth_gain_db = np.interp(relative_frequency, knot_positions, knot_gains_db)

    # Tilt part: a straight line through 0 dB at mid-band with a random slope.
    tilt_db = np.random.uniform(-6, 6)
    tilt_gain_db = tilt_db * (relative_frequency - 0.5)

    gain_db = smooth_gain_db + tilt_gain_db
    equalised_spectrum = spectrum * 10 ** (gain_db / 20)
    return np.fft.irfft(equalised_spectrum, n=len(y)).astype(np.float32)
