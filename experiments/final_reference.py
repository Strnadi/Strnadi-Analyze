# %% [markdown]
# # final_reference
#
# The reference recipe on the final label table.
#
# **Question.** The baseline of the final series: the recipe of `strnadnet_audited_labels`, unchanged, on the
# labels after all five audit rounds. Three seeds give the noise every other final run is judged against.
#
# **Differs from `strnadnet_audited_labels` in:** nothing; only the label table is the final one.
#
# **Status:** final series (2026-11-25): trained on the final label table (audit rounds 1-5) with
# three seeds, `STRNADI_SEED=0|1|2`.
#
# Run from the repository root with `python -m experiments.final_reference`.
# Training writes to `$STRNADI_WS/runs/final_reference/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'final_reference'

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
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
