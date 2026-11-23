# %% [markdown]
# # r7_visual_qa_fixes
#
# R7: R2 + the fixes found by looking at spectrograms: strong clips re-cut around the whole song
# (2,740 windows moved), counterfactual truncations filled with spectrally matched background
# (cf_v4), SNR measured in the song band.
#
# **Question.** Many R2 "errors" were clips starting mid-song or zero-padded; clean windows and
# clean synthetic data beat R2.
#
# **Status:** queued (from the research session). Not finished on 2026-11-23. Audit caveats for
# reading it: its evaluation windows differ from R2's, so the two are only comparable after re-
# scoring R2 on the aligned windows (evaluation/compare_runs.py now refuses mixed tables); ~19
# moved "Unfinished" windows may now contain an ending the labeller never saw; BirdNET-app clips
# stay unaligned, so window placement correlates with class.
#
# Run from the repository root with `python -m experiments.r7_visual_qa_fixes`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/r7_visual_qa_fixes/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'r7_visual_qa_fixes'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v3.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_audiomoth.npy',
    eval_window='start',
    # protocol
    checkpoint_rule='inner_argmax',
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
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
