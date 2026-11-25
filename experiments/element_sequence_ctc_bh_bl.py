# %% [markdown]
# # element_sequence_ctc_bh_bl
#
# The CTC element sequence with the two whistles of BhBl / BlBh named directly: `Bh` and `Bl`.
#
# **Question.** In `element_sequence_ctc` the first whistle is `B` and the second `L` or `H`
# (lower / higher than the B before). Here BhBl is `phrase Bh Bl` and BlBh is `phrase Bl Bh`, the
# names the ornithologists use. The information is the same; does the sequence read from the
# model get better or worse? (CTC announces a token after it has heard the element, so naming
# the first whistle by what follows it costs CTC less than it may cost a per-frame tagger; see
# `element_frames_bh_bl_tags`.)
#
# **Differs from `element_sequence_ctc` in:** `element_grammar='direct'`
# (`losses.element_ctc.STRNADI_GRAMMAR_DIRECT`).
#
# **Status:** queued (2026-11-25).
#
# Run from the repository root with `python -m experiments.element_sequence_ctc_bh_bl`.
# Training writes to `$STRNADI_WS/runs/element_sequence_ctc_bh_bl/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from losses.element_ctc import TOKENS
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'element_sequence_ctc_bh_bl'

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
    element_grammar='direct',
)


# %% Model
def make_model():
    return StrnadNet(head='factorized', element_tokens=len(TOKENS))


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
