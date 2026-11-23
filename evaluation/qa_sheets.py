"""Visual QA contact sheets of exactly what the model sees (LogMel 1.5-11 kHz, 128 bins, median-normalised, 4 s).
sheets: strong (per class), weak, jcu, cf (counterfactual surgery), aug (training augmentations of the same clips),
errors (R2 OOF confusions), jcu_pred (JCU test songs by predicted class).  Usage: qa_sheets.py <sheet> [seed]"""
import sys, os, numpy as np, pandas as pd, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from types import SimpleNamespace
from training.data import load_seg
from config import CLASSES, WS
from layers.registry import build_frontend
from augment.pipeline import WaveAugment
torch.set_num_threads(2); FE = build_frontend(); N = 128000; OUT = f'{WS}/figs/qa'
sheet = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0; rng = np.random.default_rng(seed)
rd = lambda p: pd.read_csv(p, keep_default_na=False, na_values=[''])


def mel(y):
    y = y / max(np.abs(y).max(), 1e-6) * 0.5
    with torch.no_grad(): return FE(torch.from_numpy(y[None].astype(np.float32)))[0, 0].numpy()


def grid(items, name, ncol=4):
    n = len(items); nr = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nr, ncol, figsize=(4.2 * ncol, 2.3 * nr), squeeze=False)
    for a in ax.ravel(): a.axis('off')
    for a, (S, title) in zip(ax.ravel(), items):
        a.imshow(S, origin='lower', aspect='auto', cmap='magma', vmin=np.percentile(S, 30), vmax=np.percentile(S, 99.7))
        a.set_title(title, fontsize=8); [a.axvline(x, color='w', lw=0.4, ls=':') for x in (100, 200, 300)]
    plt.tight_layout(); p = f'{OUT}/{name}.png'; plt.savefig(p, dpi=55); print(p)


def clip(r, t0=None):
    t0 = r.t0 if t0 is None else t0
    return load_seg(r.npy, t0, t0 + 4, N)


if sheet == 'strong':
    d = rd(f'{WS}/data/orig_index.csv'); d = d[~d.aug]
    it = [(mel(clip(r)), f'{r.label} {r.src} {os.path.basename(r.npy)[:18]}') for c in CLASSES for r in d[d.label == c].sample(3, random_state=seed).itertuples()]
    grid(it, f'strong_s{seed}', ncol=3)
elif sheet == 'weak':
    d = rd(f'{WS}/data/weak_v3.csv'); d = d.sample(16, random_state=seed)
    grid([(mel(clip(r)), f'{r.label[:22]} {r.source} w{r.w:.1f}') for r in d.itertuples()], f'weak_s{seed}')
elif sheet == 'jcu':
    d = rd(f'{WS}/data/jcu_train.csv'); s = d[d.label != 'None'].sample(12, random_state=seed); nn_ = d[d.label == 'None'].sample(4, random_state=seed)
    grid([(mel(clip(r)), f'{r.dev} yh{r.yh:.2f} {r.label[:4]}') for r in pd.concat([s, nn_]).itertuples()], f'jcu_s{seed}')
elif sheet == 'cf':
    d = rd(f'{WS}/data/cf_v4.csv'); d['song'] = d.name.str.split('_', n=2).str[2]
    songs = rng.choice(d.song.unique(), 4, replace=False); it = []
    for s in songs:
        for v in ['orig', 'swap', 'trunc2', 'trunc12']:
            r = d[(d.song == s) & (d.variant == v)].iloc[0]; it.append((mel(clip(r)), f'{v} -> {r.label}  {s[:14]}'))
    grid(it, f'cf4_s{seed}')
elif sheet == 'aug':
    d = rd(f'{WS}/data/orig_index.csv'); d = d[~d.aug & d.label.isin(['BhBl', 'BE', 'BlBh'])].sample(4, random_state=seed)
    nb = np.load(f'{WS}/data/noise_bank_audiomoth.npy', allow_pickle=True)
    base = dict(speed=0.0, p_far=0.0, p_noise=0.0, snr=(0, 20), p_codec=0.0, p_splice=0.0)
    FAR = f'{WS}/data/farfield_stats.csv'
    variants = [('clean', {}), ('noise 0-20dB', dict(p_noise=1.0)), ('noise 0-2dB', dict(p_noise=1.0, snr=(0, 2))), ('far-field', dict(p_far=1.0)),
                ('codec', dict(p_codec=1.0)), ('splice x4', dict(p_splice=1.0)), ('speed', dict(speed=0.02))]
    it = []
    for r in d.itertuples():
        for nm, kw in variants:
            np.random.seed(seed + len(it)); aug = WaveAugment(base | {'snr_mode': 'ending'} | kw, noise_bank=list(nb), far_csv=FAR)
            it.append((mel(aug(clip(r))), f'{nm} | {r.label}'))
    grid(it, f'aug_s{seed}', ncol=7)
elif sheet == 'far':
    d = rd(f'{WS}/data/orig_index.csv'); d = d[~d.aug & d.label.isin(['BhBl', 'BlBh'])].sample(3, random_state=seed)
    nb = np.load(f'{WS}/data/noise_bank_audiomoth.npy', allow_pickle=True); it = []
    for r in d.itertuples():
        it.append((mel(clip(r)), f'clean | {r.label}'))
        for j in range(5):
            np.random.seed(seed * 100 + len(it)); aug = WaveAugment(dict(speed=0.0, p_far=1.0, snr_mode='ending'), noise_bank=list(nb), far_csv=f'{WS}/data/farfield_stats.csv')
            it.append((mel(aug(clip(r))), f'far-field v2 #{j} | {r.label}'))
    grid(it, f'far_s{seed}', ncol=6)
elif sheet in ('align', 'unf_cut'):
    o = rd(f'{WS}/data/orig_index.csv'); A = rd(f'{WS}/data/orig_aligned.csv')
    st = ['start_cut'] if sheet == 'align' else ['end_cut_unf_kept']
    sel = A[A.align_status.isin(st)].sample(8, random_state=seed); it = []
    for i in sel.index:
        r0, r1 = o.loc[i], A.loc[i]
        it.append((mel(load_seg(r0.npy, r0.t0, r0.t0 + 4, N)), f'ORIGINAL {r0.label} {r0["name"][:16]}'))
        src = r1.npy if sheet == 'align' else [p for p in [A.loc[i, 'npy']]][0]
        if sheet == 'align':
            it.append((mel(load_seg(r1.npy, r1.t0, r1.t0 + 4, N)), f'ALIGNED {r1.label} t0={r1.t0:.1f}'))
        else:  # show the source context: 4 s starting 1 s after the clip start (does the song end later?)
            import glob as g_; F = r0['name'].split('.')[0].upper()
            sp = [q for q in g_.glob(f'{WS}/audio32k/ncs/*.npy') if os.path.basename(q).split('.')[0].upper() == F][0]
            it.append((mel(load_seg(sp, r1.src_off + 2.0, r1.src_off + 6.0, N)), f'SOURCE +2..+6 s  song {r1.song_s0 - r1.src_off:.1f}-{r1.song_s1 - r1.src_off:.1f}s'))
    grid(it, f'{sheet}_s{seed}', ncol=4)
elif sheet == 'errors':
    import glob
    acc = {}
    for f in glob.glob(f'{WS}/runs/r2_s0/fold*/oof.npz'):
        z = np.load(f, allow_pickle=True); acc.update({int(i): p for i, p in zip(z['idx'], z['p'])})
    d = rd(f'{WS}/data/orig_index.csv').loc[sorted(acc)]; P = np.stack([acc[i] for i in d.index]); d['pred'] = np.array(CLASSES)[P.argmax(1)]; d['conf'] = P.max(1)
    e = d[d.pred != d.label]; it = []
    for c in ['BhBl', 'BlBh', 'None', 'BE']:
        for r in e[e.label == c].sample(min(4, (e.label == c).sum()), random_state=seed).itertuples():
            it.append((mel(clip(r)), f'true {r.label} -> {r.pred} ({r.conf:.2f}) {os.path.basename(r.npy)[:14]}'))
    grid(it, f'errors_r2_s{seed}')
elif sheet == 'jcu_pred':
    from training.inference import TorchEnsemble
    d = rd(f'{WS}/data/jcu_test.csv'); d = d[d.label != 'None'].sample(400, random_state=seed)
    X = np.stack([clip(r) for r in d.itertuples()]); X = (X / np.maximum(np.abs(X).max(1, keepdims=True), 1e-6) * 0.5).astype(np.float32)
    P = TorchEnsemble(f'{WS}/runs/r2_s0')(X); d['pred'] = np.array(CLASSES)[P.argmax(1)]; d['conf'] = P.max(1)
    print(d.pred.value_counts(normalize=True).round(3).to_dict())
    it = []
    for c, k in [('Unfinished', 8), ('BhBl', 4), ('None', 2), ('BC', 2)]:
        for j, r in enumerate(d[d.pred == c].head(k).itertuples()):
            it.append((mel(X[d.index.get_loc(r.Index)]), f'pred {c} ({r.conf:.2f}) {r.dev} yh{r.yh:.2f}'))
    grid(it, f'jcu_pred_s{seed}')
