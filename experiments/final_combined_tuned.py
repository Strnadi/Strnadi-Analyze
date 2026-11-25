# %% [markdown]
# # final_combined_tuned
#
# The combined recipe with the optimisation settings of the hyper-parameter search winner.
#
# **Question.** On the reference recipe the search winner made 116 errors against 127-147 for the
# three seeds of the reference settings, and it was also ahead on the two folds the search never
# saw. The combined recipe gained about as much by changing the training signal. Do the two gains
# add up?
#
# **Differs from `final_combined` in:** learning rate 0.005 (0.002), weight decay 0.1 (0.05),
# up to 40 epochs (25) with early stopping after 8 epochs without improvement, batch 96 (48),
# weight-averaging decay 0.995 (0.998), speed jitter 4 % (2 %), dropout 0.3 (0.2). The noise
# settings stay those of the combined recipe (the search tuned the noise of the reference recipe,
# which is mixed at a level set by the whole band).
#
# **Status:** queued (2026-11-25).
#
# Run from the repository root with `python -m experiments.final_combined_tuned`.
# Training writes to `$STRNADI_WS/runs/final_combined_tuned/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'final_combined_tuned'

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
    noise_probability=0.5,
    snr_reference='ending',
    snr_db_range=(6.0, 26.0),
    speed_jitter=0.04,
    splice_probability=0.3,
    treble_cut_probability=0.25,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized', drop=0.3)


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
