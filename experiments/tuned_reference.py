# %% [markdown]
# # tuned_reference
#
# The winner of the hyper-parameter search (`tools/tune_hyperparameters.py`, trial 15), on all folds.
#
# **Question.** The search scored 16 random settings on folds 0-2; the best reached 0.969 there
# against 0.965 for the reference settings, a difference the size of the seed noise. Folds 3 and 4
# were never seen by the search: does the winner also beat the reference on them?
#
# **Differs from `final_reference` in:** learning rate 0.005 (0.002), weight decay 0.1 (0.05),
# 40 epochs (25), batch 96 (48), weight-averaging decay 0.995 (0.998), noise on 30 % of the clips
# (50 %) at 5-25 dB (0-20), speed jitter 4 % (2 %), dropout 0.3 (0.2), early stopping after 8
# epochs without improvement.
#
# **Status:** queued (2026-11-25).
#
# Run from the repository root with `python -m experiments.tuned_reference`.
# Training writes to `$STRNADI_WS/runs/tuned_reference/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'tuned_reference'

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
    epochs=40,
    samples_per_epoch=6144,
    batch_size=96,
    learning_rate=0.005,
    weight_decay=0.1,
    ema_decay=0.995,
    early_stopping_patience=8,
    # waveform augmentation
    noise_probability=0.3,
    snr_reference='band',
    snr_db_range=(5.0, 25.0),
    speed_jitter=0.04,
    splice_probability=0.3,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized', drop=0.3)


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
