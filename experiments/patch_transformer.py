# %% [markdown]
# # patch_transformer
#
# Architecture comparison: small patch transformer (3.1 M) from scratch on the R7 recipe.
#
# **Question.** Can attention over patches beat convolution here?
#
# **Status:** queued (from the research session). Not finished. Both reviews expect it to lose:
# 3,744 labelled clips are too few for a transformer from scratch. Low priority.
#
# Run from the repository root with `python -m experiments.patch_transformer`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/runs/patch_transformer/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.patch_transformer import PatchTransformer
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'patch_transformer'

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
    folds=[0, 1, 2],
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
    return PatchTransformer(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation)
cross_validate(RUN, make_model, settings)

# %% Evaluate on the held-out clips of every fold
summarize_run(RUN)
plot_run(RUN)
plot_error_examples(RUN)
