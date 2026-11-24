# %% [markdown]
# # element_sequence_ctc
#
# The reference model with a second output: the sequence of song elements, trained with CTC.
#
# **Question.** "Phrase + X without B" called XB is the largest remaining confusion: a pooled
# "is the song complete?" score does not say which element is missing. Does an explicit element
# sequence (phrase, B, C, E, L, H, X; `losses/element_ctc.py`), learned from the clip labels alone,
# make the classifier better, and can the decoded sequence itself classify?
# (Suggested by the project owner on 2026-11-24.)
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `element_tokens=8` on the model,
# `sequence_loss_weight=0.5`. The sequence loss uses hand-labelled clips only: with the weak
# candidate sets included, the decoder read most BC / BE songs as "phrase B" (first run:
# sequence accuracy 0.89, see docs/RESULTS.md).
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.element_sequence_ctc`.
# Training writes to `$STRNADI_WS/runs/element_sequence_ctc/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from losses.element_ctc import TOKENS
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'element_sequence_ctc'

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
)


# %% Model
def make_model():
    return StrnadNet(head='factorized', element_tokens=len(TOKENS))


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
