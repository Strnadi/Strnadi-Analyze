"""Tests of the losses, augmentations, sampler and every optional path of the trainer, on a tiny synthetic
workspace (random audio, random labels): no real data needed.

    python -m tests.test_training

The trainer runs are smoke tests (does each loss / data path execute and write its outputs); they say nothing
about accuracy.
"""
import json
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix='strnadi_test_')
os.environ['STRNADI_WS'] = _TMP                       # must be set before `config` is imported

import numpy as np
import pandas as pd
import torch

from config import CLASSES, SR
from training.defaults import make_config
from training.trainer import train

TINY = dict(widths=(8, 8, 16, 16), d=32, emb_dim=32)


def make_workspace():
    rng = np.random.default_rng(0)
    for d in ('audio', 'data', 'emb', 'runs', 'results'):
        os.makedirs(f'{_TMP}/{d}', exist_ok=True)
    recs = []
    for r in range(12):
        p = f'{_TMP}/audio/rec{r}.npy'; np.save(p, (rng.normal(0, 2000, 12 * SR)).astype(np.int16)); recs.append(p)
    strong, weak, unl = [], [], []
    for i in range(56):
        r = i % 12; lab = CLASSES[i % 7]; t0 = float(rng.uniform(0, 6))
        strong.append(dict(name=f'c{i}', label=lab, npy=recs[r], t0=t0, t1=t0 + 4.4, grp=f'F{r}', fold=r % 2, aug=False, src='ncs' if i % 5 else 'birdnet_mobile',
                           song_t0=t0 + 0.6, song_t1=t0 + 3.0, elements=json.dumps([['phrase', t0 + 0.6, t0 + 2.2], ['B', t0 + 2.3, t0 + 2.7]]),
                           m_f1=4.8, m_f2=5.2, m_f3=5.6, m_df=0.4, m_bw=0.4, m_t2=0.3, m_has_first=1.0, m_has_second=float(i % 2)))
    for i in range(60):
        r = i % 12; t0 = float(rng.uniform(0, 6)); d = ['BC', 'BE', 'BhBl', 'BlBh', 'XB'][r % 5]
        weak.append(dict(name=f'w{i}', label='None' if i % 10 == 0 else f'{d}|Unfinished|None', npy=recs[r], t0=t0, t1=t0 + 4.4, fold=r % 2, aug=False,
                         w=1.0, sw=1.0, source='ncs' if r % 3 else 'vizina', rec=f'ncs:F{r}', song_t0=t0 + 0.6, song_t1=t0 + 3.0))
        unl.append(dict(name=f'u{i}', npy=recs[r], t0=t0, t1=t0 + 4.4, fold=r % 2, rec=f'ncs:F{r}', source='ncs'))
    pd.DataFrame(strong).to_csv(f'{_TMP}/data/orig_index.csv', index=False)
    pd.DataFrame(weak).to_csv(f'{_TMP}/data/weak.csv', index=False); pd.DataFrame(unl).to_csv(f'{_TMP}/data/unl.csv', index=False)
    bank = np.empty(4, dtype=object); bank[:] = [rng.normal(0, 0.05, 6 * SR).astype(np.float16) for _ in range(4)]
    np.save(f'{_TMP}/data/noise.npy', bank, allow_pickle=True)
    np.savez(f'{_TMP}/emb/orig_perch.npz', emb=rng.normal(size=(56, 1536)).astype(np.float16), spatial=rng.normal(size=(56, 16, 1536)).astype(np.float16))


def run(name, **kw):
    cfg = make_config(dict(name=name, folds=[0], epochs=2, epoch_size=16, bs=8, threads=1, workers=0, **kw))
    res = train(cfg); out = f'{_TMP}/runs/{name}/fold0'
    assert os.path.exists(f'{out}/model.pt') and os.path.exists(f'{out}/oof.npz') and not os.path.exists(f'{out}/resume.pt')
    z = np.load(f'{out}/oof.npz', allow_pickle=True)
    assert z['p'].shape == (len(z['y']), 7) and np.allclose(z['p'].sum(1), 1, atol=1e-4)
    return res


def test_trainer_all_paths():
    make_workspace()
    run('everything', model='strnadnet', model_kw=dict(head='factorized', **TINY), extra=['data/weak.csv'], pool_share={'weak': 0.4},
        unlabeled='data/unl.csv', bags=dict(table='data/weak.csv', k=3, n_bags=2), w_bag=0.5, w_frames=1.0, w_distill=0.5, teacher='cache',
        noise='data/noise.npy', p_noise=0.5, p_splice=0.5, p_eq=0.5, mixstyle_p=1.0, select='inner_smooth3')
    run('flat_head_partial_labels', model='strnadnet', model_kw=TINY, extra=['data/weak.csv'], margin=0.0, select='inner_argmax')
    run('no_inner_val', model='strnadnet', model_kw=TINY, inner_val=0.0, select='last')
    run('cosine_margin', model='strnadnet', model_kw=dict(head='cosine', head_kw=dict(k=2), **TINY), margin=0.2)
    run('student_of_teacher', model='strnadnet', model_kw=TINY, teacher='online', teacher_run='everything', w_distill=0.5, w_kd=1.0)
    run('perch_spatial', model='perch_temporal', model_kw=dict(precomputed=True, d=32), inputs='perch_spatial')
    run('perch_pooled', model='perch_mlp', model_kw=dict(precomputed=True, hidden=32), inputs='perch_pooled')
    run('elements', model='element_tagger', model_kw=dict(widths=(8, 8, 16, 16), d=32, emb_dim=32), w_elements=1.0)
    run('measure', model='strnadnet_measure', model_kw=TINY, w_measure=0.5)
    run('wave_input_model', model='ending_zoom', model_kw=dict(coarse_widths=(8, 8, 16, 16), zoom_widths=(8, 16, 16), d=32, emb_dim=32))


def test_resume():
    cfg = make_config(dict(name='resume', folds=[0], epochs=1, epoch_size=16, bs=8, threads=1, workers=0, model='strnadnet', model_kw=TINY))
    train(cfg); d = f'{_TMP}/runs/resume/fold0'
    # pretend the run was killed after epoch 0 of a 2-epoch schedule: put a resume file back, drop the outputs
    from training import trainer as T
    orig_remove = os.remove
    os.remove(f'{d}/oof.npz'); cfg2 = dict(cfg, epochs=2); kept = {}
    def keep(p):
        (kept.setdefault('p', p) if p.endswith('resume.pt') else orig_remove(p))
    os.remove = keep
    try:
        T.train_fold(dict(cfg, name='resume_a', epochs=1), 0)        # leaves resume.pt of a finished 1-epoch run behind
    finally:
        os.remove = orig_remove
    os.makedirs(d, exist_ok=True); os.replace(f'{_TMP}/runs/resume_a/fold0/resume.pt', f'{d}/resume.pt')
    T.train_fold(cfg2, 0)
    log = json.load(open(f'{d}/log.json'))
    assert [r['ep'] for r in log] == [0, 1], log                    # epoch 0 came from the resume file, only epoch 1 was trained


def test_losses():
    from losses.bag_complete import bag_complete_nll
    from losses.partial_label import partial_label_nll
    from losses.soft_cross_entropy import soft_cross_entropy
    from losses.nt_xent import nt_xent
    from losses.frame_bce import frame_bce
    from layers.factorized_head import FactorizedHead
    out = FactorizedHead(16)(torch.randn(6, 16))
    one = bag_complete_nll(out, torch.tensor([2] * 6), 1)           # a bag of one song: -log p(song, complete, dialect)
    ref = -(out['log_song'] + out['log_complete'] + out['log_dialect'][:, 2])
    assert torch.allclose(one, ref, atol=1e-4)
    assert (bag_complete_nll(out, torch.tensor([2, 2]), 3) <= one.view(2, 3).min(1).values + 1e-5).all()   # more songs can only help the bag
    logits = torch.randn(4, 7); hard = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3]), 7).float()
    assert torch.allclose(partial_label_nll(logits, hard), soft_cross_entropy(logits, hard), atol=1e-5)      # singleton set = CE
    assert (partial_label_nll(logits, torch.ones(4, 7)).abs() < 1e-5).all()                                 # full set = no information
    z = torch.randn(8, 16); assert nt_xent(z, z + 0.01 * torch.randn(8, 16)) < nt_xent(z, torch.randn(8, 16))
    assert frame_bce(torch.randn(2, 100), torch.full((2, 200), -1.0)) == 0


def test_augmentations():
    from augment.speed_perturb import speed_perturb
    from augment.null_splice import null_splice
    from augment.noise_mix import mix_noise, signal_power, noise_power
    from augment.pipeline import WaveAugment
    rng = np.random.default_rng(0); np.random.seed(0)
    y = (rng.normal(0, 0.01, 128000)).astype(np.float32); t = np.arange(SR) / SR
    y[40000:40000 + SR] += 0.5 * np.sin(2 * np.pi * 5000 * t).astype(np.float32)              # loud "phrase"
    y[80000:80000 + SR // 2] += 0.05 * np.sin(2 * np.pi * 4500 * t[:SR // 2]).astype(np.float32)   # faint "terminal element", 20 dB lower
    nz = rng.normal(0, 0.1, 128000).astype(np.float32)
    assert len(speed_perturb(y)) == len(y) and len(null_splice(y)) == len(y)
    assert signal_power(y, 'ending') < 0.05 * signal_power(y, 'band')                           # the ending reference sees the faint element
    for mode in ('clip', 'band', 'ending'):
        added = mix_noise(y, nz, 10.0, mode) - y
        got = 10 * np.log10(signal_power(y, mode) / noise_power(added, mode))
        assert abs(got - 10.0) < 0.5, (mode, got)
    out = WaveAugment(dict(p_noise=1.0, p_splice=1.0, p_eq=1.0), noise_bank=[nz])(y)
    assert out.shape == y.shape and abs(np.abs(out).max() - 0.5) < 1e-4 and out.dtype == np.float32


def test_sampler():
    from training.sampler import pool_sampling_weights
    df = pd.DataFrame(dict(label=['BC'] * 50 + ['Unfinished'] * 400 + ['BC|Unfinished|None'] * 900, pool=['strong'] * 450 + ['weak'] * 900))
    w, shares = pool_sampling_weights(df, {'weak': 0.35})
    assert abs(shares['weak'] - 0.35) < 1e-6 and abs(w.sum() - 1) < 1e-6
    assert w[:50].sum() / w[50:450].sum() > 50 / 400                                            # rare class is up-weighted inside its pool


if __name__ == '__main__':
    torch.set_num_threads(1)
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f(); print('ok', k, flush=True)
