# %% [markdown]
# # backbone_efficientnet_b0_imagenet
#
# ImageNet-pretrained efficientnet_b0 as the 2-D backbone (4.7 M parameters).
#
# **Question.** Does a pretrained image backbone beat the from-scratch StrnadNet when both see clean labels? The backbone gets a tenth of the learning rate of the new layers.
#
# **Differs from `strnadnet_audited_labels` (the reference) in:** `TimmTemporal(backbone='efficientnet_b0')`, `backbone_lr_factor=0.1`.
#
# **Status:** done (from this session, 2026-11-24).
#
# Run from the repository root with `python -m experiments.backbone_efficientnet_b0_imagenet`.
# Training writes to `$STRNADI_WS/runs/backbone_efficientnet_b0_imagenet/` (TensorBoard: `tb/` in every fold directory).

# %% Imports
from evaluation.report import plot_run, summarize_run
from model.timm_temporal import TimmTemporal
from training.fit import cross_validate
from training.settings import TrainSettings

# %% Settings (everything not listed keeps its default from training/settings.py)
RUN = 'backbone_efficientnet_b0_imagenet'

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
    backbone_lr_factor=0.1,
    # waveform augmentation
    noise_probability=0.5,
    snr_reference='band',
    splice_probability=0.3,
)


# %% Model
def make_model():
    return TimmTemporal(backbone='efficientnet_b0', head='factorized')


# %% Train: one model per fold (recording-grouped cross-validation), then evaluate
if __name__ == '__main__':
    cross_validate(RUN, make_model, settings)
    summarize_run(RUN)
    plot_run(RUN)
