from .recipes import R7

EXPERIMENT = dict(
    R7,
    description='R7: R2 + the fixes found by looking at spectrograms: strong clips re-cut around the whole song (2,740 windows '
                'moved), counterfactual truncations filled with spectrally matched background (cf_v4), SNR measured in the song band.',
    hypothesis='Many R2 "errors" were clips starting mid-song or zero-padded; clean windows and clean synthetic data beat R2.',
    origin='research-session', status='queued',
    result='Not finished on 2026-11-23. Audit caveats for reading it: its evaluation windows differ from R2\'s, so the two are '
           'only comparable after re-scoring R2 on the aligned windows (evaluation/compare_runs.py now refuses mixed tables); '
           '~19 moved "Unfinished" windows may now contain an ending the labeller never saw; BirdNET-app clips stay unaligned, '
           'so window placement correlates with class.',
)
