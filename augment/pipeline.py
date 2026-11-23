"""The training-time waveform augmentation chain, configured from the experiment dict.

Order: speed -> far-field -> noise -> other species -> codec -> null splice -> polarity -> peak normalise.
Peak normalisation (to 0.5) is applied to every clip, training and inference alike.
"""
import numpy as np
from .speed_perturb import speed_perturb
from .far_field import far_field, load_profiles
from .noise_mix import mix_noise, draw_chunk
from .species_mix import species_mix
from .codec_roundtrip import codec_roundtrip
from .null_splice import null_splice
from .random_eq import random_eq


def peak_normalize(y, peak=0.5):
    pk = np.abs(y).max()
    return (y / pk * peak).astype(np.float32) if pk > 1e-6 else y.astype(np.float32)


class WaveAugment:
    def __init__(self, cfg, noise_bank=None, species_bank=None, far_csv=None):
        g = cfg.get
        self.speed, self.p_noise, self.snr, self.snr_mode = g('speed', 0.02), g('p_noise', 0.0), g('snr', (0.0, 20.0)), g('snr_mode', 'ending')
        self.p_far, self.p_codec, self.p_splice, self.p_species, self.p_eq = g('p_far', 0.0), g('p_codec', 0.0), g('p_splice', 0.0), g('p_species', 0.0), g('p_eq', 0.0)
        self.noise, self.species = noise_bank, species_bank
        self.profiles = load_profiles(far_csv) if (far_csv and self.p_far > 0) else []

    def __call__(self, y, force_noise=False):
        n = len(y); r = np.random.rand
        if self.speed > 0 and r() < 0.5:
            y = speed_perturb(y, self.speed)
        if self.p_eq > 0 and r() < self.p_eq:
            y = random_eq(y)
        far_snr = None
        if self.profiles and r() < self.p_far:
            y, far_snr = far_field(y, self.profiles[np.random.randint(len(self.profiles))])
        if self.noise is not None and (far_snr is not None or force_noise or r() < self.p_noise):
            snr = far_snr if far_snr is not None else np.random.uniform(*self.snr)
            y = mix_noise(y, draw_chunk(self.noise, n), snr, self.snr_mode)
        if self.species is not None and r() < self.p_species:
            y = species_mix(y, draw_chunk(self.species, n))
        if self.p_codec > 0 and r() < self.p_codec:
            y = codec_roundtrip(y)
        if self.p_splice > 0 and r() < self.p_splice:
            y = null_splice(y)
        if r() < 0.5:
            y = -y
        return peak_normalize(y)
