# %% [markdown]
# # learning_curve_50pct_with_weak_songs
#
# Learning curve: the reference trained on 50 % of the hand-labelled recordings, with
# the weakly labelled and synthetic songs as in the reference.
#
# **Question.** How does accuracy depend on the number of hand-labelled clips, and how much of a
# small labelled set can the 50 thousand songs with only a file-level label make up for? This is
# the question a new project on the platform faces: how many clips must an expert label before the
# classifier is useful. The test folds are the full ones in every run.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `labelled_fraction=0.5`.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.learning_curve_50pct_with_weak_songs`.
# Training writes to `$STRNADI_WS/runs/learning_curve_50pct_with_weak_songs/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'learning_curve_50pct_with_weak_songs'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    labelled_fraction=0.5,
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
    snr_reference='band',
    splice_probability=0.3,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
