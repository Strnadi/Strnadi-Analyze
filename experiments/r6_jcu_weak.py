# %% [markdown]
# # r6_jcu_weak
#
# R6: R2 + 38k weak songs {BhBl, Unfinished, None} from 28 JCU AudioMoth recorders (passive, far-
# field).
#
# **Question.** Passive-recorder data makes the model robust in that domain.
#
# **Status:** running (from the research session). Fold 0 only: macro-F1 0.775 (R2 fold 0: 0.799).
# Review warnings: (1) every passive-recorder training song is constrained to BhBl, so "passive
# acoustics => BhBl" is a learnable shortcut, and the JCU "wrong-dialect rate" metric rewards it;
# this is the main risk for the Italian data. (2) JCU test recorders sit 120-170 m from training
# recorders. Check with evaluation/dialect_prior_shift.py before trusting it.
#
# Run from the repository root with `python -m experiments.r6_jcu_weak`, or step through the cells
# in an editor / Jupyter whose working directory is the repository root. Training writes to
# `$STRNADI_WS/runs/r6_jcu_weak/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'r6_jcu_weak'

settings = TrainSettings(
    # data
    labelled_table='data/orig_index.csv',
    extra_tables={
        'data/weak_v3.csv': 0.3,
        'data/cf_v3.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
        'data/jcu_train.csv': 0.08,
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
    snr_reference='clip',
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
