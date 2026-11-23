from .recipes import R7

EXPERIMENT = dict(
    R7, p_far=0.3,
    description='R3: R7 + far-field simulation (distance-scaled treble loss, reverberation, measured SNR) on 30 % of clips.',
    hypothesis='Robustness on passive recorders (JCU, Italy) without hurting phone recordings.',
    origin='research-session', status='queued',
    result='Not run. The v1 augmentation was unphysical (random tilt sign); v2 is in augment/far_field.py.',
    needs='data/farfield_stats.csv (processing: BioDCASE 2025 Task 3 playback statistics)',
)
