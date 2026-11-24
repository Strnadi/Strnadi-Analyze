# %% [markdown]
# # strnadnet_field_ready_190k
#
# The field-ready recipe on the smallest network (0.19 M parameters): the phone-size candidate.
#
# **Question.** Each of these was tested alone against the reference and cost nothing measurable
# on the archive test set; do they still cost nothing together? They are wanted for reasons this
# test set cannot show: the machine-labelled songs add 25k BirdNET-app detections (phones, other
# countries), the augmentations imitate distance, other species and devices, and the element
# sequence makes the answer readable ("phrase X, no B") and flags uncertain clips where it
# disagrees with the class head.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** machine-labelled songs as a
# training table (`expanded_machine_labels`), far-field / other-species / device-EQ augmentation
# (`augment_real_world`), element-sequence output with CTC (`element_sequence_ctc`).
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.strnadnet_field_ready_190k`.
# Training writes to `$STRNADI_WS/runs/strnadnet_field_ready_190k/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from losses.element_ctc import TOKENS
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'strnadnet_field_ready_190k'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/songs_machine_labelled.csv': 0.25,
        'data/weak_v5.csv': 0.20,
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
    # auxiliary losses
    sequence_loss_weight=0.5,
)


# %% Model
def make_model():
    return StrnadNet(
        widths=(8, 16, 32, 64), d=64, emb_dim=64, head='factorized', element_tokens=len(TOKENS)
    )


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
