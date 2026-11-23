# %% [markdown]
# # legacy_densenet121
#
# The production recipe (good-model-2 lineage) as it was, ported to PyTorch: magma mel -> ImageNet
# DenseNet121 -> patch attention pooling -> MLP; AdamW 1e-4, wide SpecAugment, inverse-frequency
# class balance, strong labels only.
#
# **Question.** Reproduce the deployed model under a leak-free protocol, as the reference the new
# models are compared with.
#
# **Status:** proposed (from the original Keras repository). The Keras original reported ~82.5 %
# accuracy / 64.5 % macro recall on a random by-file split (optimistic: clips of one recording and
# pre-baked augmented copies crossed the split). The deployed model on held-out Vizina males: BhBl
# recall 0.55 over 58 males, ~45 % of complete BhBl songs called Unfinished. Not yet re-trained in
# this form. Note the 27-bin / 0.8 s masks can erase a whole terminal element; kept only to
# reproduce the original.
#
# Run from the repository root with `python -m experiments.legacy_densenet121`, or step through
# the cells in an editor / Jupyter whose working directory is the repository root. Training writes
# to `$STRNADI_WS/runs/legacy_densenet121/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.densenet121_attnpool import DenseNet121AttnPool
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'legacy_densenet121'

settings = TrainSettings(
    # data
    labelled_table='data/orig_index.csv',
    eval_window='start',
    # protocol
    checkpoint_rule='inner_argmax',
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
    return DenseNet121AttnPool(pretrained=True)


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
