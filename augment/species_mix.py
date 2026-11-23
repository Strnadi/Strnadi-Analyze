"""Mix another species' vocalisation (a hard-negative window: BirdNET hears another bird, no Yellowhammer)
into the background of a song, 5-20 dB below the song's ending.

Occlusion maps showed the model leaning on context outside the song, more so when it is wrong (59 % of
importance outside the song for errors vs 48 % for correct predictions), with other birds and second songs
driving errors. This teaches it to ignore them. The label is unchanged.
Origin: research session backlog ("R8 candidate"), proposed after the Grad-CAM / occlusion analysis.
"""
import numpy as np
from .noise_mix import mix_noise


def species_mix(y, other, below_db=None):
    below_db = np.random.uniform(5, 20) if below_db is None else below_db
    return mix_noise(y, other, below_db, mode='ending')
