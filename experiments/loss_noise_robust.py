# %% [markdown]
# # loss_noise_robust
#
# The reference trained with a noise-robust loss on the hand-labelled clips.
#
# **Question.** After three audit rounds some labels are still wrong (two of my own relabels look
# wrong on a second view, and a few unreviewed XB clips show no B). Cross-entropy lets a
# confidently contradicted label dominate its batch; generalised cross-entropy bounds that loss.
# Does it help, or does it only slow down learning of the hard, correct clips?
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `robust_loss_q=0.7`
# (`losses/generalized_cross_entropy.py`); label smoothing does not apply to this loss.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.loss_noise_robust`.
# Training writes to `$STRNADI_WS/runs/loss_noise_robust/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'loss_noise_robust'

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
    # loss
    robust_loss_q=0.7,
)


# %% Model
def make_model():
    return StrnadNet(head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
