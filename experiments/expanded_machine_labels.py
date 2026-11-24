# %% [markdown]
# # expanded_machine_labels
#
# The reference plus the songs labelled by the models themselves.
#
# **Question.** 75k detected songs carry no human label. `processing/machine_label_songs.py` lets the
# cross-validated models label them and keeps a label only when it is confident, allowed by the
# recording's file-level label, and confirmed by the decoded element sequence. Does training on
# those labels beat the reference? (Each fold reads the labels written by teachers that never
# saw that fold, so the comparison is fair.) Asked for by the project owner on 2026-11-24.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** the extra table `data/songs_machine_labelled.csv` with a
# quarter of the training draws (taken from the labelled clips and the weak pool).
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.expanded_machine_labels`.
# Training writes to `$STRNADI_WS/runs/expanded_machine_labels/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'expanded_machine_labels'

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
