# %% [markdown]
# # augment_real_world
#
# The reference with the augmentations that imitate field conditions switched on.
#
# **Question.** Far-field simulation (distance-dependent treble loss and reverberation, from the
# BioDCASE playback measurements), other species singing in the background, and random device
# equalisation prepare a model for phones and passive recorders. The archive test set contains
# little of that, so the question here is only whether they cost accuracy on it; if not, they
# go into the shipped model.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `far_field_probability=0.2`, `species_probability=0.3` with
# `data/species_bank.npy` (1,500 windows of other species), `random_eq_probability=0.3`.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.augment_real_world`.
# Training writes to `$STRNADI_WS/runs/augment_real_world/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'augment_real_world'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    species_bank='data/species_bank.npy',
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
    far_field_probability=0.2,
    species_probability=0.3,
    random_eq_probability=0.3,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
