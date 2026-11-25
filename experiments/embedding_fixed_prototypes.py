# %% [markdown]
# # embedding_fixed_prototypes
#
# The combined recipe with the embedding pulled towards fixed class directions.
#
# **Question.** The map of the embedding (UMAP) puts the dialects on separate islands, but where
# the islands lie changes with every training, so a region an expert outlined would not survive a
# retraining. Here a cosine-margin loss (the ArcFace / CosFace family, with the class centres
# frozen as constants of the code) pulls every hand-labelled clip towards the direction of its
# class (`losses/fixed_prototypes.py`). Two seeds: do the classes land in the same places, is the
# fixed 2-D read-out a usable map, and what does the extra loss cost in accuracy?
#
# **Differs from `final_combined` in:** `prototype_loss_weight=0.5`. Evaluated with
# `evaluation/embedding_stability.py`.
#
# **Status:** queued (2026-11-25).
#
# Run from the repository root with `python -m experiments.embedding_fixed_prototypes`.
# Training writes to `$STRNADI_WS/runs/embedding_fixed_prototypes/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'embedding_fixed_prototypes'

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
    # auxiliary losses
    prototype_loss_weight=0.5,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
