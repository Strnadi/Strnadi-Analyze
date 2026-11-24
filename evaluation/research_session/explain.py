"""What does a trained model look at?  (out-of-fold clips only: fold-k model on fold-k clips)

maps  <run> [n_per_class=30]
  Grad-CAM (last 2-D conv stage, StrnadNet `cnn` / timm feature map), occlusion sensitivity (16 mel bins x 0.2 s
  patches filled with the clip's per-bin median) and the model's own temporal attention.  For factorised heads the
  occlusion is also computed for the two decisions separately: complete-vs-unfinished logit and the dialect logit.
  Song region from the tonal segmenter; per-class shares of positive importance: inside the song, on its ending
  (last 0.7 s), outside the song, below 2.5 kHz (background band), window edges (first/last 0.2 s).
  -> results/explain_<run>.json + figs/explain/<run>_<class>.png (spectrogram | Grad-CAM | occlusion | attention)
patch <run[,run...]>
  shortcut test for the cf_v3 black-patch artefact: complete, correctly classified dialect songs get 0.5 s right
  after the song end attenuated by -12 dB / -20 dB (5 ms fades) vs. an untouched control -> change of P(Unfinished)
  and of the predicted class.  A model that learned 'dark patch = Unfinished' shifts; one that reads the song does not.
context <run[,run...]>
  context dependence: OOF song clips (all errors + equal number of correct, dialect/Unfinished labels) re-scored with
  (a) everything outside [onset-0.15, end+POST] replaced by the clip's own background-matched noise ('song only') and
  (b) the song replaced by background ('context only').  Accuracy of each variant, and how many errors are fixed /
  correct clips broken by removing the context -> context-driven errors (other birds, second songs) vs. song-driven.
  POST (env CTX_POST, default 1.2 s) must cover the terminal element: the segmenter's song end is often the B
  element, and BC's C / BE's E follow it by 0.3-1.1 s.  With POST = 0.35 the test wrongly showed R2 leaning on context.
  -> results/explain_context_<run>.{csv,json} + figs/explain/<run>_context.png (control | song only | context only)
"""

import os, sys, json, glob, numpy as np, pandas as pd, torch, torch.nn.functional as F, matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import CLASSES, C2I, WS
from training.data import load_seg, background_noise
from training.checkpoint import load_logits_model as load_torch_model
from processing.segment import segment_songs

torch.set_num_threads(int(os.environ.get('THREADS', 2)))
SR, N, HOP = 32000, 128000, 320
DIA = ['BC', 'BE', 'BhBl', 'BlBh', 'XB']
OUT = f'{WS}/figs/explain'
os.makedirs(OUT, exist_ok=True)
# seconds kept after the detected song end in the context test (terminal C / E elements follow B by 0.3-1.1 s)
POST = float(os.environ.get('CTX_POST', 1.2))


def norm(y):
    return (y / max(np.abs(y).max(), 1e-6) * 0.5).astype(np.float32)


def song_span(y, faint=True):
    segs, _ = segment_songs(y, hi=10, lo=4, max_gap=0.3, min_dur=0.3, max_dur=6.0)
    ev = []
    for s0, s1, _ in segs:
        if ev and s0 - ev[-1][1] < 0.6 and s1 - ev[-1][0] <= 4.0:
            ev[-1][1] = s1
        else:
            ev.append([s0, s1])
    ev = [e for e in ev if e[1] - e[0] >= 0.3]
    if not ev:
        return None
    # union of tonal events (the longest one often missed phrase or ending)
    s0, s1 = min(e[0] for e in ev), max(e[1] for e in ev)
    # faint short terminal elements (BC's C: quiet ~0.2 s whistle 0.3-1 s after B) are below hi / min_dur: extend the
    # end with low-threshold elements that follow within 1.0 s.  This catches only some of them (research log 17.4).
    if faint:
        fs, _ = segment_songs(y, hi=6, lo=3, max_gap=0.1, min_dur=0.06, max_dur=1.0)
        for a_, b_, _ in sorted(fs):
            if s1 - 0.05 < a_ <= s1 + 1.0 and b_ - s0 <= 4.0:
                s1 = max(s1, b_)
    return s0, s1


def fmodel_parts(m):
    """(spectrogram -> 2-D conv map) hook target and frontend for StrnadNet / TimmNet"""
    if hasattr(m, 'cnn'):
        return m.cnn
    return m.body if hasattr(m, 'body') else None


def targets(m, S):
    """dict of scalar targets per item: 'pred' = log p(pred class); for factorised heads also 'complete' (logit)
    and 'dialect' (log q of the predicted dialect)"""
    out = m(S, return_all=True)
    t = {
        'logp': torch.log_softmax(out['logits'], 1)
    }  # the factorised head returns log-probs already (idempotent)
    if 'log_complete' in out:
        lc = out['log_complete']
        t['complete'] = lc - torch.log1p(-lc.exp().clamp(max=1 - 1e-6))
        t['lq'] = out['log_dialect']
    return t


def gradcam(m, S, cls):
    mod = fmodel_parts(m)
    store = {}
    h = mod.register_forward_hook(lambda _m, _i, o: store.__setitem__('a', o))
    S = S.clone().requires_grad_(False)
    m.zero_grad()
    out = torch.log_softmax(m(S), 1)
    a = store['a']
    h.remove()
    g = torch.autograd.grad(out[torch.arange(len(S)), cls].sum(), a)[0]
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=S.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    return (cam / cam.amax((1, 2), keepdim=True).clamp_min(1e-9)).detach().numpy()


def occlusion(m, S1, fb=16, ft=20, kinds=('pred',), cls=None):
    """S1 (1,1,F,T) -> {kind: (F,T) importance = target(full) - target(occluded), >=0 part}"""
    Fq, T = S1.shape[-2:]
    med = S1.median(-1, keepdim=True).values
    boxes = [(f0, t0) for f0 in range(0, Fq, fb) for t0 in range(0, T, ft)]
    X = S1.repeat(len(boxes) + 1, 1, 1, 1)
    for j, (f0, t0) in enumerate(boxes, 1):
        X[j, :, f0 : f0 + fb, t0 : t0 + ft] = med[0, :, f0 : f0 + fb].expand(
            -1, -1, min(ft, T - t0)
        )
    with torch.no_grad():
        res = {}
        for i in range(0, len(X), 64):
            t = targets(m, X[i : i + 64])
            for k in kinds:
                if k == 'pred':
                    v = t['logp'][:, cls]
                elif k == 'complete' and 'complete' in t:
                    v = t['complete']
                elif k == 'dialect' and 'lq' in t:
                    v = t['lq'].max(1).values
                else:
                    continue
                res.setdefault(k, []).append(v)
    maps = {}
    for k, v in res.items():
        v = torch.cat(v).numpy()
        M = np.zeros((Fq, T), np.float32)
        for j, (f0, t0) in enumerate(boxes, 1):
            M[f0 : f0 + fb, t0 : t0 + ft] = max(0.0, v[0] - v[j])
        maps[k] = M
    return maps


def shares(M, span, fcent):
    tot = M.sum() + 1e-12
    T = M.shape[1]
    t = (np.arange(T) + 0.5) * HOP / SR
    r = {
        'lowband': float(M[fcent < 2500].sum() / tot),
        'edges': float(M[:, (t < 0.2) | (t > 3.8)].sum() / tot),
    }
    if span is not None:
        s0, s1 = span
        r['song'] = float(M[:, (t >= s0 - 0.1) & (t <= s1 + 0.1)].sum() / tot)
        r['ending'] = float(M[:, (t >= s1 - 0.7) & (t <= s1 + 0.1)].sum() / tot)
        r['outside'] = 1 - r['song']
        ts = float(((t >= s0 - 0.1) & (t <= s1 + 0.1)).mean())
        r['song_conc'] = r['song'] / max(ts, 1e-3)  # >1 = focused on the song
        te = float(((t >= s1 - 0.7) & (t <= s1 + 0.1)).mean())
        r['ending_conc'] = r['ending'] / max(te, 1e-3)
    return r


def oof_rows(run, n_per, seed=0):
    d = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
    d = d[~d.aug]
    acc = {}
    for f in glob.glob(f'{WS}/runs/{run}/fold*/oof.npz'):
        z = np.load(f, allow_pickle=True)
        k = int(f.split('fold')[1].split('/')[0])
        acc.update({int(i): (k, p) for i, p in zip(z['idx'], z['p'])})
    d = d.loc[[i for i in d.index if i in acc]].copy()
    d['k'] = [acc[i][0] for i in d.index]
    d['pred'] = [CLASSES[int(acc[i][1].argmax())] for i in d.index]
    return pd.concat(
        [g.sample(min(n_per, len(g)), random_state=seed) for _, g in d.groupby('label')]
    )


def run_maps(run, n_per):
    rows = oof_rows(run, n_per)
    stats = []
    models = {}
    fcent = None
    figs = {}
    for r in rows.itertuples():
        if r.k not in models:
            models[r.k] = load_torch_model(f'{WS}/runs/{run}/fold{r.k}/model.pt')
        m = models[r.k]
        if fcent is None:
            fe = m.frontend
            fcent = (fe.bank.fb.argmax(1).numpy() + fe.stft.bin_lo) * SR / fe.stft.n_fft
        y = norm(load_seg(r.npy, r.t0, r.t0 + 4, N))
        span = song_span(y)
        with torch.no_grad():
            S = m.frontend(torch.from_numpy(y[None]))
        ci = C2I[r.pred]
        cam = gradcam(m, S, torch.tensor([ci]))[0]
        occ = occlusion(m, S, kinds=('pred', 'complete', 'dialect'), cls=ci)
        with torch.no_grad():
            att = m(S, return_all=True).get('att')
            att = att[0].numpy() if att is not None else None
        st = dict(label=r.label, pred=r.pred, correct=r.label == r.pred, has_song=span is not None)
        for k, M in [('cam', cam)] + [(f'occ_{k}', v) for k, v in occ.items()]:
            st.update({f'{k}_{q}': v for q, v in shares(M, span, fcent).items()})
        stats.append(st)
        figs.setdefault(r.label, [])
        if len(figs[r.label]) < 4:
            figs[r.label].append(
                (
                    S[0, 0].numpy(),
                    cam,
                    occ['pred'],
                    att,
                    span,
                    f'{r.label}->{r.pred} {os.path.basename(r.npy)[:16]}',
                )
            )
    D = pd.DataFrame(stats)
    for lab, items in figs.items():
        fig, ax = plt.subplots(len(items), 4, figsize=(16, 2.4 * len(items)), squeeze=False)
        for row, (S, cam, occ, att, span, title) in zip(ax, items):
            lo, hi = np.percentile(S, 30), np.percentile(S, 99.7)
            row[0].imshow(S, origin='lower', aspect='auto', cmap='magma', vmin=lo, vmax=hi)
            row[0].set_title(title, fontsize=8)
            for a_, M, nm in [(row[1], cam, 'Grad-CAM'), (row[2], occ, 'occlusion (pred)')]:
                a_.imshow(S, origin='lower', aspect='auto', cmap='gray', vmin=lo, vmax=hi)
                a_.imshow(M, origin='lower', aspect='auto', cmap='jet', alpha=0.45)
                a_.set_title(nm, fontsize=8)
            if att is not None:
                row[3].plot(np.linspace(0, 4, len(att)), att)
                row[3].set_xlim(0, 4)
                row[3].set_title('temporal attention', fontsize=8)
            if span is not None:
                for a_ in row[:3]:
                    a_.axvline(span[0] * SR / HOP, color='w', lw=0.8, ls='--')
                    a_.axvline(span[1] * SR / HOP, color='w', lw=0.8, ls='--')
                row[3].axvspan(span[0], span[1], color='orange', alpha=0.2)
        plt.tight_layout()
        plt.savefig(f'{OUT}/{run}_{lab}.png', dpi=55)
        plt.close()
    cols = [c for c in D.columns if c.startswith(('cam_', 'occ_'))]
    summ = D[D.has_song].groupby('label')[cols].mean().round(3)
    print(summ.T.to_string())
    print(
        'correct-vs-wrong (occ_pred_outside):',
        D[D.has_song].groupby('correct').occ_pred_outside.mean().round(3).to_dict()
        if 'occ_pred_outside' in D
        else '',
    )
    json.dump(
        {'per_class': summ.to_dict(), 'n': int(len(D))},
        open(f'{WS}/results/explain_{run}.json', 'w'),
        indent=1,
    )


def run_patch(runs):
    d = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
    d = d[~d.aug & d.label.isin(DIA)]
    res = {}
    for run in runs:
        acc = {}
        for f in glob.glob(f'{WS}/runs/{run}/fold*/oof.npz'):
            z = np.load(f, allow_pickle=True)
            k = int(f.split('fold')[1].split('/')[0])
            acc.update({int(i): (k, p) for i, p in zip(z['idx'], z['p'])})
        sel = [
            i for i in d.index if i in acc and CLASSES[int(acc[i][1].argmax())] == d.at[i, 'label']
        ]
        rows = d.loc[sel].sample(min(300, len(sel)), random_state=0)
        out, pred, models = {}, {}, {}
        for i, r in rows.iterrows():
            y = norm(load_seg(r.npy, r.t0, r.t0 + 4, N))
            sp = song_span(y)
            if sp is None or sp[1] + 1.2 > 3.95:
                continue
            fade = 160
            X = [y]
            names = ['control']
            for dist in (0.05, 0.35, 0.65):
                a, b = int((sp[1] + dist) * SR), int((sp[1] + dist + 0.5) * SR)
                for gdb in (12, 20):
                    g = 10 ** (-gdb / 20)
                    e = np.ones(N, np.float32)
                    e[a:b] = g
                    e[a - fade : a] = np.linspace(1, g, fade)
                    e[b : b + fade] = np.linspace(g, 1, fade)
                    X.append(y * e)
                    names.append(f'd{dist}_m{gdb}')
            k = acc[i][0]
            if k not in models:
                models[k] = load_torch_model(f'{WS}/runs/{run}/fold{k}/model.pt')
            with torch.no_grad():
                P = torch.softmax(
                    models[k](torch.from_numpy(np.stack(X).astype(np.float32))), 1
                ).numpy()
            for j, v in enumerate(names):
                out.setdefault(v, []).append(P[j, C2I['Unfinished']])
                pred.setdefault(v, []).append(CLASSES[int(P[j].argmax())])
        n = len(out['control'])
        r_ = {
            'n': n,
            **{f'P_unf_{v}': float(np.mean(out[v])) for v in out},
            **{
                f'flip_to_unf_{v}': float(np.mean([p == 'Unfinished' for p in pred[v]]))
                for v in out
            },
            **{
                f'flip_any_{v}': float(np.mean([p != c for p, c in zip(pred[v], pred['control'])]))
                for v in out
                if v != 'control'
            },
        }
        print(run, json.dumps(r_), flush=True)
        res[run] = r_
    json.dump(res, open(f'{WS}/results/explain_patch_{"+".join(runs)}.json', 'w'), indent=1)


def strong_table(run):
    """the labelled table a run was evaluated on (stored with its out-of-fold predictions)"""
    for f in sorted(glob.glob(f'{WS}/runs/{run}/fold*/oof.npz')):
        z = np.load(f, allow_pickle=True)
        if 'strong' in z.files and str(z['strong']):
            p = str(z['strong'])
            return p if os.path.isabs(p) else f'{WS}/{p}'
    args_path = f'{WS}/runs/{run}/args.json'
    a = json.load(open(args_path)) if os.path.exists(args_path) else {}
    return a.get('strong') or f'{WS}/data/orig_index.csv'


def splice_bg(y, a, b, keep_inside, fade=640):
    """replace [a, b) (keep_inside=False) or everything outside it (True) by background-matched noise, 20 ms fades"""
    bg = background_noise(y, len(y))
    m = np.zeros(len(y), np.float32)
    a, b = max(0, a), min(len(y), b)
    m[a:b] = 1.0
    if a > 0:
        m[max(0, a - fade) : a] = np.linspace(0, 1, a - max(0, a - fade))
    if b < len(y):
        m[b : min(len(y), b + fade)] = np.linspace(1, 0, min(len(y), b + fade) - b)
    if not keep_inside:
        m = 1 - m
    return (y * m + bg * (1 - m)).astype(np.float32)


def run_context(runs):
    res = {}
    for run in runs:
        d = pd.read_csv(strong_table(run), keep_default_na=False, na_values=[''])
        d = d[~d.aug & d.label.isin(DIA + ['Unfinished'])]
        acc = {}
        for f in glob.glob(f'{WS}/runs/{run}/fold*/oof.npz'):
            z = np.load(f, allow_pickle=True)
            k = int(f.split('fold')[1].split('/')[0])
            acc.update({int(i): (k, p) for i, p in zip(z['idx'], z['p'])})
        d = d.loc[[i for i in d.index if i in acc]].copy()
        d['pred'] = [CLASSES[int(acc[i][1].argmax())] for i in d.index]
        wrong = d[d.pred != d.label]
        right = d[d.pred == d.label].sample(
            min(len(wrong), (d.pred == d.label).sum()), random_state=0
        )
        rows = pd.concat([wrong, right])
        models, recs, sheet = {}, [], []
        for i, r in rows.iterrows():
            y = norm(load_seg(r.npy, r.t0, r.t0 + 4, N))
            sp = song_span(y)
            if sp is None:
                continue
            a, b = int((sp[0] - 0.15) * SR), int((sp[1] + POST) * SR)
            # control | song only (context replaced) | context only (song replaced)
            X = np.stack([y, splice_bg(y, a, b, True), splice_bg(y, a, b, False)])
            k = acc[i][0]
            if k not in models:
                models[k] = load_torch_model(f'{WS}/runs/{run}/fold{k}/model.pt')
            with torch.no_grad():
                P = torch.softmax(models[k](torch.from_numpy(X)), 1).numpy()
            pr = [CLASSES[int(p.argmax())] for p in P]
            interesting = r.pred == r.label and (pr[1] != r.label or pr[2] == r.label)
            if len(sheet) < 12 and interesting and r.label in ('BC', 'BlBh', 'BE'):
                with torch.no_grad():
                    mels = [models[k].frontend(torch.from_numpy(x[None]))[0, 0].numpy() for x in X]
                title = f'{r.label}: ctl {pr[0]} | song-only {pr[1]} | ctx-only {pr[2]}'
                sheet.append((mels, sp, title))
            recs.append(
                dict(
                    idx=i,
                    label=r.label,
                    orig_correct=r.pred == r.label,
                    ctl=pr[0] == r.label,
                    song_only=pr[1] == r.label,
                    ctx_only_none=pr[2] == 'None',
                    ctx_only_same=pr[2] == r.label,
                    pred_ctl=pr[0],
                    pred_song=pr[1],
                )
            )
        D = pd.DataFrame(recs)
        W, R = D[~D.orig_correct], D[D.orig_correct]
        os.makedirs(f'{WS}/results', exist_ok=True)
        D.to_csv(f'{WS}/results/explain_context_{run}.csv', index=False)
        if sheet:
            fig, ax = plt.subplots(len(sheet), 3, figsize=(12, 2.2 * len(sheet)), squeeze=False)
            for row, (Ss, sp, title) in zip(ax, sheet):
                lo, hi = np.percentile(Ss[0], 30), np.percentile(Ss[0], 99.7)
                for a_, S_, nm in zip(row, Ss, ['control', 'song only', 'context only']):
                    a_.imshow(S_, origin='lower', aspect='auto', cmap='magma', vmin=lo, vmax=hi)
                    a_.axis('off')
                    a_.axvline((sp[0] - 0.15) * SR / HOP, color='c', lw=0.8)
                    a_.axvline((sp[1] + POST) * SR / HOP, color='c', lw=0.8)
                    a_.set_title(f'{nm}: {title}' if nm == 'control' else nm, fontsize=7)
            plt.tight_layout()
            plt.savefig(f'{OUT}/{run}_context.png', dpi=55)
            plt.close()
        print(
            'broken correct -> song-only pred:',
            R[~R.song_only].groupby('label').pred_song.value_counts().to_dict(),
        )
        r_ = {
            'n_wrong': len(W),
            'n_right': len(R),
            'errors_fixed_by_song_only': float(W.song_only.mean()),
            'correct_broken_by_song_only': float(1 - R.song_only.mean()),
            'ctx_only_pred_none': float(D.ctx_only_none.mean()),
            'ctx_only_still_label_right': float(R.ctx_only_same.mean()),
            'per_class_fixed': W.groupby('label').song_only.mean().round(3).to_dict(),
            'per_class_broken': (1 - R.groupby('label').song_only.mean()).round(3).to_dict(),
        }
        print(run, json.dumps(r_), flush=True)
        res[run] = r_
    json.dump(res, open(f'{WS}/results/explain_context_{"+".join(runs)}.json', 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'maps':
        run_maps(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    elif sys.argv[1] == 'context':
        run_context(sys.argv[2].split(','))
    else:
        run_patch(sys.argv[2].split(','))
