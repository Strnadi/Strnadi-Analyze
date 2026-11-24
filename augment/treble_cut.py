"""Remove everything above ~6 kHz, and change the label when that removes the song's ending.

The terminal elements X, C and E lie at 6.5-9 kHz; the phrase, the B whistle and the second whistle
of BhBl lie below 6 kHz. Cutting the treble of an XB, BC or BE song therefore leaves "phrase + B":
by the project's rule an Unfinished song. Distance does the same to a real recording (air and
vegetation absorb treble first), so this is both a counterfactual and a far-field simulation with
a known correct answer.

Why: with the treble removed the trained models still call 81 % of XB songs "XB". They recognise
the dialect from the pause before a low B, without seeing the X, and 15 of the 64 clips that every
model gets wrong are exactly "phrase, pause, low B, no visible X" labelled Unfinished. Training on
treble-cut XB songs as Unfinished makes the X itself the evidence.

So that "no treble" does not become a cue for Unfinished on its own, the same cut is applied to
BhBl, Unfinished and None clips with their label unchanged. BlBh clips are left alone: their
second whistle sits around 6-7 kHz, so the cut would damage it only partly.
"""

import numpy as np

#: Classes whose terminal element is above the cut: a treble-cut clip of these is "Unfinished".
ENDING_ABOVE_CUT = ('XB', 'BC', 'BE')
#: Classes whose label does not depend on the treble: cut, label unchanged.
LABEL_UNCHANGED = ('BhBl', 'Unfinished', 'None')


def treble_cut(waveform, cutoff_hz, sr=32000, roll_off_hz=500.0):
    """Low-pass a waveform: full level below `cutoff_hz`, nothing above `cutoff_hz + roll_off_hz`."""
    spectrum = np.fft.rfft(waveform)
    frequency = np.fft.rfftfreq(len(waveform), 1 / sr)
    gain = np.clip((cutoff_hz + roll_off_hz - frequency) / roll_off_hz, 0, 1)
    return np.fft.irfft(spectrum * gain, n=len(waveform)).astype(np.float32)


def treble_cut_example(waveform, label):
    """Cut the treble of a labelled clip.

    Returns:
        (waveform, label): the cut waveform and the label that is true of it, or the inputs
        unchanged when the clip's class is not one the cut applies to.
    """
    if label in ENDING_ABOVE_CUT:
        return treble_cut(waveform, np.random.uniform(5800, 6200)), 'Unfinished'
    if label in LABEL_UNCHANGED:
        return treble_cut(waveform, np.random.uniform(5800, 6200)), label
    return waveform, label
