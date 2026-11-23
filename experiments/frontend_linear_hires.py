# %% [markdown]
# # frontend_linear_hires
#
# Front-end ablation: linear frequency axis, 256 bins over 2-9.7 kHz (30 Hz per bin), 64 ms
# window.
#
# **Question.** Dialect criteria are 100-250 Hz bandwidths and 1000-2000 Hz offsets, measured by
# experts at 86 Hz resolution; the default mel axis has 60-70 Hz bins and the stem halves that at
# once.
#
# **Status:** proposed (from the research session). Not run (in the research experiment matrix and
# literature report, never queued).
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.frontend_linear_hires`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/frontend_linear_hires/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'frontend_linear_hires'

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
    return StrnadNet(
        head='factorized',
        frontend=dict(
            bank='linear', norm='percentile', n_bins=256, fmin=2000.0, fmax=9680.0, n_fft=2048
        ),
        stem_stride=(2, 2),
    )


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
