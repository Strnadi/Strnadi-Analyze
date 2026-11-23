from .recipes import CLEAN, CLEAN_NEEDS, with_model_kw

EXPERIMENT = dict(
    with_model_kw(CLEAN, frontend=dict(norm='percentile', n_fft=[512, 1024, 4096])),
    description='Front-end ablation: three STFT window lengths (16 / 32 / 128 ms) stacked as input channels.',
    hypothesis='The missed element is a long, thin, faint tone: a 128 ms window integrates it and gains several dB of SNR, while '
               'the short window keeps the fast syllables of the first phrase. Targets complete-vs-Unfinished AUC for BhBl '
               '(0.79-0.83) and BlBh (0.84-0.89) directly.',
    origin='review-fork', status='proposed', result='Not run.', needs=CLEAN_NEEDS,
)
