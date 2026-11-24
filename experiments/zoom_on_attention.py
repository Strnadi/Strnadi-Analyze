# %% [markdown]
# # zoom_on_attention
#
# Look twice: a coarse network over the whole window, plus a high-resolution (27 Hz, 5 ms) network
# on the 1.2 s around the moment where the coarse network's own attention peaks.
#
# **Question.** `ending_zoom` centres its crop with an energy rule that stops at the B whistle and
# misses the faint C / E element. The trained models' temporal attention peaks on the last element
# of the song (the service finds songs with it). Does a second, sharper look at that place settle
# the borderline endings: a short, low, falling C against the flat second whistle of BlBh, a
# slightly rising E, a faint second whistle?
#
# **Differs from `ending_zoom` in:** `locate='attention'`, crop 0.7 s before to 0.5 s after.
#
# **Status:** done (from this session, 2026-11-24).
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.zoom_on_attention`, or step through the cells
# in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/zoom_on_attention/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.ending_zoom import EndingZoomNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'zoom_on_attention'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    eval_window='center',
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
    return EndingZoomNet(locate='attention', before_s=0.7, after_s=0.5)


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
