# %% [markdown]
# # contour_ridge
#
# Pitch-contour TCN on a sub-bin ridge track (parabolic peak interpolation on a linear 39 Hz
# axis).
#
# **Question.** The research contour model quantises pitch to 60-70 Hz mel bins; Hz-accurate
# contours let a 0.27 M-parameter 1-D model apply the published criteria.
#
# **Status:** proposed (from review proposal). Not run.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.contour_ridge`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/contour_ridge/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.contour_tcn import ContourTCN
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'contour_ridge'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v3.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    eval_window='center',
    # protocol
    checkpoint_rule='inner_smooth3',
    # optimisation
    epochs=25,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.002,
    # waveform augmentation
    noise_probability=0.5,
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
)


# %% Model
def make_model():
    return ContourTCN(
        features='ridge',
        head='factorized',
        frontend=dict(
            bank='linear', norm='percentile', n_bins=192, fmin=2000.0, fmax=9500.0, n_fft=2048
        ),
    )


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
