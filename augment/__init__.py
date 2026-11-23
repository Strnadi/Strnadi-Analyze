"""Waveform augmentations (numpy, one per file) and `pipeline.WaveAugment`, which applies them in a fixed order.

The rule for every augmentation here: it must not change the label. Dialects are defined by Hz offsets of
100-250 Hz and by element ORDER, so there is no large pitch shift, no time reversal, no time-stretching, no
mixing of two dialects, no reflect padding (reversed audio turns BhBl into a BlBh-like pattern), and noise must
not bury the faint terminal element.
"""
