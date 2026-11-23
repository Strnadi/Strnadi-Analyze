# %% [markdown]
# # fixed_schedule_no_inner_val
#
# Fixed 12-epoch schedule, EMA weights, last epoch; no inner validation set.
#
# **Question.** Checkpoint selection on a ~380-clip inner set is noise (+-0.02 per epoch, as large
# as the effects being measured): over 18 finished folds last-epoch scored 0.760 vs 0.757 for the
# argmax rule. Dropping it returns 15 % of the training recordings (the set has 79 BlBh clips),
# and held-out F1 is flat from epoch 4-8, so 25 epochs are about twice as long as needed.
#
# **Status:** proposed (from review proposal). Not run. The outer-fold curve is still logged every
# epoch, so overfitting stays visible; it is just not used to pick a checkpoint.
#
# **Needs.** processing/clean_noise_bank.py (-> data/noise_bank_clean.npy); data/orig_aligned.csv;
# data/cf_v4.csv
#
# Run from the repository root with `python -m experiments.fixed_schedule_no_inner_val`, or step
# through the cells in an editor / Jupyter whose working directory is the repository root.
# Training writes to `$STRNADI_WS/runs/fixed_schedule_no_inner_val/`.

# %% Imports
from evaluation.report import plot_error_examples, plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'fixed_schedule_no_inner_val'

settings = TrainSettings(
    # data
    labelled_table='data/orig_aligned.csv',
    extra_tables={
        'data/weak_v5.csv': 0.35,
        'data/cf_v4.csv': 0.08,
        'data/vizann_v1.csv': 0.07,
    },
    noise_bank='data/noise_bank_clean.npy',
    eval_window='center',
    # protocol
    inner_validation_share=0.0,
    checkpoint_rule='last',
    # optimisation
    epochs=12,
    samples_per_epoch=6144,
    batch_size=48,
    learning_rate=0.002,
    # waveform augmentation
    noise_probability=0.5,
    snr_db_range=(6.0, 26.0),
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
