# %% [markdown]
# # legacy_mobilenetv3_delta
#
# Legacy low-latency variant, ported: mel + delta + delta-delta channels -> ImageNet
# MobileNetV3-Small -> patch attention pooling -> 128 -> classes.
#
# **Question.** An edge-friendly backbone for the phone at the production recipe's accuracy.
#
# **Status:** rerun (from the original Keras repository). Re-targeted on 2026-11-24 to the audited clips and
# best-validation-loss checkpoints, 5 folds; results in docs/RESULTS.md. No stored result in the legacy repo.
# Not re-trained.
#
# Run from the repository root with `python -m experiments.legacy_mobilenetv3_delta`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/runs/legacy_mobilenetv3_delta/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.mobilenetv3_delta import MobileNetV3Delta
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'legacy_mobilenetv3_delta'

settings = TrainSettings(
    # data
    labelled_table='data/clips_audited.csv',
    eval_window='center',
    # optimisation
    epochs=30,
    samples_per_epoch=4096,
    batch_size=32,
    learning_rate=0.0001,
    weight_decay=0.01,
    label_smoothing=0.0,
    class_balance_power=1.0,
    # waveform augmentation
    speed_jitter=0.0,
    # spectrogram augmentation
    freq_mask_width=27,
    time_mask_width=80,
)


# %% Model
def make_model():
    return MobileNetV3Delta(pretrained=True)


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
