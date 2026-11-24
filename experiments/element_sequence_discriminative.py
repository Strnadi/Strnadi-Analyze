# %% [markdown]
# # element_sequence_discriminative
#
# The element-sequence model trained so that the classes compete.
#
# **Question.** With plain CTC the decoded sequence was a weaker classifier than the class head
# (0.92 vs 0.95): CTC only raises the likelihood of the labelled sequence and never penalises the
# shorter reading "phrase B" of a BC song. Here each class is scored by the summed CTC likelihood
# of its sequences and a cross-entropy over the classes is added, so "phrase B C" has to beat
# "phrase B". Does the sequence then classify as well as the head?
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `element_tokens=8`,
# `sequence_loss_weight=0.5`, `sequence_class_loss_weight=0.5`.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.element_sequence_discriminative`.
# Training writes to `$STRNADI_WS/runs/element_sequence_discriminative/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from losses.element_ctc import TOKENS
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'element_sequence_discriminative'

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
    # auxiliary losses
    sequence_loss_weight=0.5,
    sequence_class_loss_weight=0.5,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized', element_tokens=len(TOKENS))


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
