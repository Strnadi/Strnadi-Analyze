"""Mix another species' vocalisation (a hard-negative window: BirdNET hears another bird, no
Yellowhammer) into the background of a song, 5-20 dB below the song's ending.

Occlusion maps showed the model leaning on context outside the song, more so when it is wrong
(59 % of importance outside the song for errors vs 48 % for correct predictions), with other birds
and second songs driving errors. This teaches it to ignore them. The label is unchanged.

Origin: research session backlog ("R8 candidate"), proposed after the Grad-CAM / occlusion
analysis.
"""

import numpy as np
from .noise_mix import mix_noise


def species_mix(y, other, below_db=None):
    """Add another species' vocalisation to a song, quieter than the song's ending.

    Args:
        y: waveform of the song clip, 1-D float32 array.
        other: waveform of the other species, same length as `y`.
        below_db: how many dB below the song's ending the other species is mixed in; None draws
            5-20 dB at random (global `np.random` state).

    Returns:
        The mixture as a float32 waveform, same length as `y`.
    """
    if below_db is None:
        below_db = np.random.uniform(5, 20)

    # "x dB below the ending" is the same as mixing noise at an SNR of x dB in 'ending' mode.
    return mix_noise(y, other, below_db, mode='ending')
