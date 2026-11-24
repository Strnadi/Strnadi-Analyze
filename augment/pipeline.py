"""The training-time waveform augmentation chain.

Order: speed -> device EQ -> far-field -> noise -> other species -> second song -> codec -> null splice -> polarity
-> peak normalisation. Peak normalisation (to 0.5) is applied to every clip, in training and at
inference alike.
"""

import numpy as np

from .codec_roundtrip import codec_roundtrip
from .far_field import far_field, load_profiles
from .noise_mix import draw_chunk, mix_noise
from .null_splice import null_splice
from .random_eq import random_eq
from .species_mix import species_mix
from .speed_perturb import speed_perturb


def peak_normalize(waveform, peak=0.5):
    """Scale a waveform so its largest absolute sample equals `peak` (silence is left untouched)."""
    current_peak = np.abs(waveform).max()
    if current_peak <= 1e-6:
        return waveform.astype(np.float32)
    return (waveform / current_peak * peak).astype(np.float32)


def happens(probability):
    """Draw a random yes/no with the given probability."""
    return np.random.rand() < probability


class WaveAugment:
    """Applies the augmentations switched on in a `TrainSettings` object to one waveform.

    Args:
        settings: the experiment's `TrainSettings`.
        noise_bank: list of background-noise chunks, or None.
        species_bank: list of other-species windows, or None.
        far_field_csv: table of measured far-field profiles (needed only for far-field simulation).
    """

    def __init__(
        self, settings, noise_bank=None, species_bank=None, far_field_csv=None, song_bank=None
    ):
        self.settings = settings
        self.noise_bank = noise_bank
        self.species_bank = species_bank
        self.song_bank = song_bank

        self.far_field_profiles = []
        if far_field_csv and settings.far_field_probability > 0:
            self.far_field_profiles = load_profiles(far_field_csv)

    def __call__(self, waveform, force_noise=False, may_add_song=False):
        """Return an augmented, peak-normalised copy of `waveform` (same length)."""
        settings = self.settings
        n_samples = len(waveform)

        if settings.speed_jitter > 0 and happens(0.5):
            waveform = speed_perturb(waveform, settings.speed_jitter)

        if happens(settings.random_eq_probability):
            waveform = random_eq(waveform)

        # Far-field simulation dictates the noise level that goes with the simulated distance.
        far_field_snr = None
        if self.far_field_profiles and happens(settings.far_field_probability):
            profile = self.far_field_profiles[np.random.randint(len(self.far_field_profiles))]
            waveform, far_field_snr = far_field(waveform, profile)

        add_noise = far_field_snr is not None or force_noise or happens(settings.noise_probability)
        if self.noise_bank is not None and add_noise:
            if far_field_snr is not None:
                snr_db = far_field_snr
            else:
                snr_db = np.random.uniform(*settings.snr_db_range)
            noise = draw_chunk(self.noise_bank, n_samples)
            waveform = mix_noise(waveform, noise, snr_db, settings.snr_reference)

        if self.species_bank is not None and happens(settings.species_probability):
            other_species = draw_chunk(self.species_bank, n_samples)
            waveform = species_mix(waveform, other_species)

        # "Mixup of two songs", kept label-safe: the added bird is quieter and its song is
        # unfinished, so the mixture still has the label of the foreground clip. The caller sets
        # `may_add_song` only for clips that contain a Yellowhammer (a "None" clip would stop
        # being "None").
        add_song = may_add_song and self.song_bank is not None
        if add_song and happens(settings.background_song_probability):
            waveform = species_mix(waveform, draw_chunk(self.song_bank, n_samples))

        if happens(settings.codec_probability):
            waveform = codec_roundtrip(waveform)

        if happens(settings.splice_probability):
            waveform = null_splice(waveform)

        if happens(0.5):
            waveform = -waveform  # polarity flip: free invariance

        return peak_normalize(waveform)
