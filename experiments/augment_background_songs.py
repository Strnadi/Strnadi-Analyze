# %% [markdown]
# # augment_background_songs
#
# The reference plus a second, quieter Yellowhammer singing in the background of 30 % of the clips.
#
# **Question.** "Mixup of two songs" was always switched off, because mixing two dialects makes
# the label undefined. A label-safe form: the added song is unfinished and 5-20 dB quieter, and it
# is never added to a "None" clip, so the label of the clip stays true. Does it help with the
# windows that hold two songs (one kind of remaining error)?
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `song_bank` (1,200 unfinished
# songs from outside the folds, `processing/build_background_song_bank.py`) and
# `background_song_probability=0.3`.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.augment_background_songs`.
# Training writes to `$STRNADI_WS/runs/augment_background_songs/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'augment_background_songs'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    song_bank='data/background_song_bank.npy',
    eval_window='center',
    # optimisation
    epochs=25,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.002,
    # waveform augmentation
    noise_probability=0.5,
    snr_reference='band',
    splice_probability=0.3,
    background_song_probability=0.3,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
