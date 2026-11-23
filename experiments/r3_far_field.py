# %% [markdown]
# # r3_far_field
#
# R3: R7 + far-field simulation (distance-scaled treble loss, reverberation, measured SNR) on 30 %
# of clips.
#
# **Question.** Robustness on passive recorders (JCU, Italy) without hurting phone recordings.
#
# **Status:** queued (from the research session). Not run. The v1 augmentation was unphysical
# (random tilt sign); v2 is in augment/far_field.py.
#
# **Needs.** data/farfield_stats.csv (processing: BioDCASE 2025 Task 3 playback statistics)
#
# Run from the repository root with `python -m experiments.r3_far_field`, or step through the
# cells in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/r3_far_field/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'r3_far_field'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
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
    far_field_probability=0.3,
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
