"""Default experiment configuration. An experiment file overrides any of these keys (see experiments/)."""
import copy
import os
from config import WS

DEFAULTS = dict(
    # ---- model
    model='strnadnet', model_kw={}, init=None,          # init: path of a pretrained encoder state (SSL / distilled)
    # ---- data (paths relative to the workspace $STRNADI_WS unless absolute)
    strong='data/orig_index.csv',                        # human-labelled clips with CV folds
    extra=[],                                            # weak / synthetic tables added to training only
    pool_share={},                                       # share of training draws per extra table (by file stem)
    unlabeled=None,                                      # table for FixMatch
    bags=None,                                           # dict(table=..., k=6, sources=('ncs',)) for the bag loss
    noise=None, species=None,                            # .npy banks for noise / other-species mixing
    inputs='wave',                                       # wave | perch_pooled | perch_spatial (cached embeddings)
    win=4.0, eval_crop='center',                         # eval window inside [t0, t1]: 'center' (corrected) or 'start' (research runs)
    # ---- protocol
    folds=[0, 1, 2, 3, 4], seed=0, inner_val=0.15,
    select='inner_smooth3',                              # checkpoint rule: inner_smooth3 | inner_argmax | last
    log_outer=True,                                      # outer-fold curve for diagnostics only, never for selection
    # ---- optimisation
    epochs=25, epoch_size=4096, bs=48, lr=2e-3, wd=0.05, backbone_lr_mult=1.0, ema=0.998, grad_clip=5.0,
    label_smoothing=0.05, balance=0.5, focal=0.0, margin=0.0, margin_scale=16.0,
    # ---- waveform augmentation (augment/pipeline.py)
    speed=0.02, p_noise=0.0, snr=(0.0, 20.0), snr_mode='ending', p_far=0.0, p_codec=0.0, p_splice=0.0, p_species=0.0, p_eq=0.0,
    # ---- spectrogram augmentation
    spec_aug=dict(freq_masks=2, freq_width=8, time_masks=2, time_width=10), mixstyle_p=0.0,
    # ---- weak / semi-supervised / auxiliary losses
    pll_eps=0.05, w_weak=1.0,                            # candidate-set escape mass for NCS rows; weight of the factorised weak loss
    w_u=1.0, u_ratio=1.0, tau_min=0.5,                   # FixMatch
    w_distill=0.0, teacher='cache', teacher_run=None,    # embedding distillation: cached Perch targets or an online torch teacher
    w_kd=0.0, kd_temperature=2.0,                        # logit distillation from `teacher_run`
    w_bag=0.0, w_frames=0.0, w_elements=0.0, w_measure=0.0,
    # ---- runtime
    threads=4, workers=2, compile=False,
)


def make_config(experiment):
    cfg = copy.deepcopy(DEFAULTS)
    unknown = set(experiment) - set(cfg) - {'name', 'description', 'hypothesis', 'origin', 'status', 'result', 'runner', 'args', 'needs'}
    if unknown:
        raise KeyError(f'unknown experiment keys: {sorted(unknown)}')
    cfg.update(copy.deepcopy(experiment))
    return cfg


def ws(path):
    return path if os.path.isabs(path) else os.path.join(WS, path)
