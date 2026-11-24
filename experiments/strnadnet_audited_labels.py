# %% [markdown]
# # strnadnet_audited_labels
#
# StrnadNet with the factorised head and the weak-label pools, trained and scored on the audited
# clip labels.
#
# **Question.** How good is the small model once the labels describe what is in the window?
# Seven earlier runs disagreed with 441 of the 3,744 clip labels. All 441 were looked at as
# spectrograms (with the source recording around the clip): 209 labels were wrong (mostly a first
# phrase plus ONE terminal element labelled as a dialect), 119 were right, 113 cannot be judged
# from the audio (ending under noise, cut by the clip edge). `data/clips_audited.csv` holds the
# corrected labels without the unjudgeable clips; `docs/LABEL_AUDIT.md` has the rules.
#
# Also new here: folds without duplicate recordings, the noise bank without Yellowhammer songs,
# checkpoints chosen by the lowest inner-validation loss.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.strnadnet_audited_labels`.
# Training writes to `$STRNADI_WS/runs/strnadnet_audited_labels/` (TensorBoard: `tb/` per fold).

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'strnadnet_audited_labels'

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
    snr_reference='band',
    splice_probability=0.3,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)

    # %% Evaluate on the held-out clips of every fold
    summarize_run(RUN)
    plot_run(RUN)
    plot_error_examples(RUN)
