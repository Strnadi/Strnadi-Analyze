# %% [markdown]
# # ending_zoom
#
# Detect-then-zoom: a coarse network over the whole window plus a high-resolution (27 Hz, 5 ms)
# network on the 1.5 s around the end of the song.
#
# **Question.** Given a complete song the dialect is right 97 % of the time; the errors are a
# missed faint ending. Resolution spent only where the ending is costs little and makes the
# 100-250 Hz criteria 4-9 bins wide.
#
# **Status:** proposed (from review proposal). Not run. The research session listed a "two-
# resolution zoom model" and never built it.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.ending_zoom`, or step through the cells
# in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/ending_zoom/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.ending_zoom import EndingZoomNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'ending_zoom'

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
    return EndingZoomNet()


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
