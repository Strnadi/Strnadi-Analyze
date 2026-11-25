# %% [markdown]
# # final_field_ready
#
# The candidate for shipping: the field-ready recipe plus the two changes of `final_combined`.
#
# **Question.** `strnadnet_field_ready` (machine-labelled songs, far-field / other-species / device-EQ augmentation,
# element-sequence output) cost about half a point on the archive test set. Does it still, with the
# ending-referenced noise and the treble-cut counterfactuals added?
#
# **Differs from `strnadnet_field_ready` in:** `snr_reference='ending'` with `snr_db_range=(6, 26)` (`augment_ending_referenced_noise`) and `treble_cut_probability=0.25` (`counterfactual_treble_cut`); the final label table.
#
# **Status:** final series (2026-11-25): trained on the final label table (audit rounds 1-5) with
# three seeds, `STRNADI_SEED=0|1|2`.
#
# Run from the repository root with `python -m experiments.final_field_ready`.
# Training writes to `$STRNADI_WS/runs/final_field_ready/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from losses.element_ctc import TOKENS
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'final_field_ready'

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
    snr_reference='ending',
    snr_db_range=(6.0, 26.0),
    splice_probability=0.3,
    treble_cut_probability=0.25,
    far_field_probability=0.2,
    species_probability=0.3,
    random_eq_probability=0.3,
    # auxiliary losses
    sequence_loss_weight=0.5,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized', element_tokens=len(TOKENS))


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
