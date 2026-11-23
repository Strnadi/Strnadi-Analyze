from .recipes import CLEAN, CLEAN_NEEDS

EXPERIMENT = dict(
    CLEAN, model='contour_tcn',
    model_kw=dict(features='ridge', head='factorized',
                  frontend=dict(bank='linear', norm='percentile', n_bins=192, fmin=2000.0, fmax=9500.0, n_fft=2048)),
    description='Pitch-contour TCN on a sub-bin ridge track (parabolic peak interpolation on a linear 39 Hz axis).',
    hypothesis='The research contour model quantises pitch to 60-70 Hz mel bins; Hz-accurate contours let a 0.27 M-parameter '
               '1-D model apply the published criteria.',
    origin='review-fork', status='proposed', result='Not run.', needs=CLEAN_NEEDS,
)
