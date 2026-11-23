# %% [markdown]
# # frontend_multires
#
# Front-end ablation: three STFT window lengths (16 / 32 / 128 ms) stacked as input channels.
#
# **Question.** The missed element is a long, thin, faint tone: a 128 ms window integrates it and
# gains several dB of SNR, while the short window keeps the fast syllables of the first phrase.
# Targets complete-vs-Unfinished AUC for BhBl (0.79-0.83) and BlBh (0.84-0.89) directly.
#
# **Status:** proposed (from review proposal). Not run.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.frontend_multires`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/frontend_multires/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'frontend_multires'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
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
        frontend=dict(norm='percentile', n_fft=[512, 1024, 4096]),
    )


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
