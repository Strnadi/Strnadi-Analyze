# %% [markdown]
# # head_cosine_margin
#
# A cosine classifier with an additive margin (CosFace) instead of the factorised head.
#
# **Question.** Margin heads (ArcFace, CosFace) were built for the identity model and proposed for
# the dialect embedding at the start of the project, but never trained on the dialect classes. Does
# a 7-class cosine head with a margin separate the classes better than the plain flat head
# (`ablation_flat_head`), and how far is it from the factorised head, which can use the weak labels
# question by question?
#
# **Differs from `final_combined` in:** `StrnadNet(head='cosine')` and `margin=0.2`.
#
# **Status:** queued (2026-11-25).
#
# Run from the repository root with `python -m experiments.head_cosine_margin`.
# Training writes to `$STRNADI_WS/runs/head_cosine_margin/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'head_cosine_margin'

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
    snr_reference='ending',
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
    treble_cut_probability=0.25,
    # cosine head
    margin=0.2,
)


# %% Model
def make_model():
    return StrnadNet(head='cosine')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
