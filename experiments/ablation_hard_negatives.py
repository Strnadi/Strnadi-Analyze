# %% [markdown]
# # ablation_hard_negatives
#
# The reference plus 6,865 windows of other species as 'None' examples.
#
# **Question.** 'None' has the lowest recall (0.79) and only 66 labelled clips. BirdNET found
# 6,865 windows where another species sings and no Yellowhammer does; does a small share of them
# (5 % of the draws; 7 % hurt the dialects in the research session) fix 'None'?
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** the extra table
# `data/hardneg_v1.csv` with 5 % of the training draws.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.ablation_hard_negatives`.
# Training writes to `$STRNADI_WS/runs/ablation_hard_negatives/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'ablation_hard_negatives'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/weak_v5.csv': 0.30,
        'data/hardneg_v1.csv': 0.05,
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
