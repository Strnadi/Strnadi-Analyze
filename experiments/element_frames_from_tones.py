# %% [markdown]
# # element_frames_from_tones
#
# Per-frame element tags trained on spans that nobody annotated.
#
# **Question.** The CTC output announces every element with one spike that carries no duration and
# is not at the element's start (median distance from the annotated onset 0.5 s). Spans built from
# the clip label and the spectrogram (`processing/build_element_spans_from_tones.py`: start of
# the first terminal element within 0.1 s of the ornithologists' time for 81 % of the held-out
# focal songs) are used here as per-frame targets: every 40 ms frame is tagged background, phrase,
# B, C, E, L, H or X. Does the model then hold each element for its length, on songs it has not
# seen, and does the sequence of runs name the right class?
#
# **Differs from `final_combined` in:** `StrnadNet(element_frames=True)`, the table
# `data/element_spans_from_tones.csv` with 15 % of the draws and `element_loss_weight=1.0`.
# Evaluated with `evaluation/element_frames.py`.
#
# **Status:** queued (2026-11-25).
#
# Run from the repository root with `python -m experiments.element_frames_from_tones`.
# Training writes to `$STRNADI_WS/runs/element_frames_from_tones/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'element_frames_from_tones'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    extra_tables={
        'data/element_spans_from_tones.csv': 0.15,
        'data/weak_v5.csv': 0.25,
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
    element_loss_weight=1.0,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized', element_frames=True)


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
