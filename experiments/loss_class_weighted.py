# %% [markdown]
# # loss_class_weighted
#
# Class balance in the loss instead of the sampler.
#
# **Question.** The reference balances classes by drawing rare ones more often. The alternative is to draw
# clips as they come and weight the loss by class. A post-hoc test said stronger balancing trades
# precision for recall; this tests the other mechanism at full strength.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `class_balance_power=0.0` (no sampling balance) and `class_weight_power=1.0`
# (loss weight inversely proportional to the class size).
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.loss_class_weighted`.
# Training writes to `$STRNADI_WS/runs/loss_class_weighted/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'loss_class_weighted'

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
    class_balance_power=0.0,
    class_weight_power=1.0,
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
