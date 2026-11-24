"""End-to-end evaluation on full recordings (review V6/V11): runs complete inference pipelines on whole files of the
Vizina held-out males and scores them against the annotated songs.

pipelines:
  songlevel  (N1, proposed): tonal event proposals -> merge close events -> 3 onset-aligned 4 s windows (TTA mean)
             -> drop events classified None -> one label per song
  prod       production runner.py logic: 4 s windows / 0.5 s hop @48 kHz TFLite, label if conf>0.7 else 'Unknown',
             merge_overlaps_simple
Scoring inside each file's annotated span [first-2 s, last+2 s]: GT song detected if a predicted YH segment covers >=50 %
of it; label = that segment's label.  Reports song recall, per-class label recall on detected songs, false alarms/min,
and per-male BhBl recall with a male-cluster bootstrap CI.
Usage: eval_e2e.py <pipeline> <model spec: run dir | tflite path> [max_files]
"""

import os, sys, json, numpy as np, pandas as pd, torch
from config import CLASSES, C2I, WS
from processing.segment import segment_songs

SR = 32000
N = 128000
DIA = ['BC', 'BE', 'BhBl', 'BlBh', 'XB']


def windows_at(y, starts):
    X = []
    for t in starts:
        i0 = max(0, min(int(t * SR), len(y) - N))
        w = y[i0 : i0 + N]
        if len(w) < N:
            w = np.concatenate(
                [w, np.random.default_rng(0).normal(0, 1e-4, N - len(w)).astype(np.float32)]
            )
        X.append(w)
    X = np.stack(X)
    return (X / np.maximum(np.abs(X).max(1, keepdims=True), 1e-6) * 0.5).astype(np.float32)


def songlevel(y, model):
    segs, _ = segment_songs(y, hi=10, lo=4, max_gap=0.3, min_dur=0.4, max_dur=6.0)
    ev = []
    for s0, s1, _ in segs:  # merge events split inside a song (gap < 0.6 s, merged length <= 4 s)
        if ev and s0 - ev[-1][1] < 0.6 and s1 - ev[-1][0] <= 4.0:
            ev[-1][1] = s1
        else:
            ev.append([s0, s1])
    ev = [e for e in ev if e[1] - e[0] >= 0.6]
    if not ev:
        return []
    offs = [-0.6, -0.4, -0.2]
    P = (
        model(windows_at(y, [e[0] + o for e in ev for o in offs]))
        .reshape(len(ev), len(offs), -1)
        .mean(1)
    )
    out = []
    for (s0, s1), p in zip(ev, P):
        lab = CLASSES[int(p.argmax())]
        if lab != 'None':
            out.append(dict(t0=s0, t1=s1, label=lab, conf=float(p.max())))
    return out


def prod(y, tfl):
    starts = np.arange(0, max(len(y) / SR - 0.01, 0.01), 0.5)
    P = tfl(windows_at(y, starts))  # TFLiteModel resamples 32->48 kHz
    labs = ['BC', 'BE', 'BhBl', 'BlBh', 'None', 'Unfinished', 'XB']
    det = []
    for t, p in zip(starts, P):
        b = int(p.argmax())
        lab = labs[b]
        if lab != 'None':
            det.append([t, t + 4.0, lab if p[b] > 0.7 else 'Unknown', float(p[b])])
    det.sort(key=lambda x: x[0])
    merged = []
    for d in det:  # merge_overlaps_simple from runner.py
        if not merged:
            merged.append(d)
            continue
        last = merged[-1]
        if last[1] + 1 > d[0]:
            if d[3] > last[3]:
                merged[-1] = d
            elif abs(d[3] - last[3]) >= 0.8:
                merged.append(d)
        else:
            merged.append(d)
    return [dict(t0=a, t1=b, label=l, conf=c) for a, b, l, c in merged]


if __name__ == '__main__':
    pipe, spec = sys.argv[1], sys.argv[2]
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    torch.set_num_threads(int(os.environ.get('THREADS', 4)))
    from training.inference import TorchEnsemble
    from evaluation.research_session.eval_external import TFLiteModel

    model = (
        TFLiteModel(spec, threads=int(os.environ.get('THREADS', 4)))
        if spec.endswith('.tflite')
        else TorchEnsemble(spec)
    )
    V = pd.read_csv(f'{WS}/data/vizina_eval.csv', dtype={'indiv': str})
    V = V[V.split == 'test']
    rows, fa, span_total = [], 0, 0.0
    for k, (npy, g) in enumerate(V.groupby('npy')):
        if k >= mx:
            break
        lo, hi = g.t0.min() - 2, g.t1.max() + 2
        a = np.load(npy, mmap_mode='r')
        i0, i1 = max(0, int(lo * SR)), min(len(a), int(hi * SR))
        y = np.asarray(a[i0:i1], np.float32) / 32768
        off = i0 / SR
        pred = {'songlevel': songlevel, 'prod': prod}[pipe](y, model)
        for p in pred:
            p['t0'] += off
            p['t1'] += off
        used = set()
        for r in g.itertuples():
            best, bo = None, 0
            for j, p in enumerate(pred):
                o = min(r.t1, p['t1']) - max(r.t0, p['t0'])
                if o / (r.t1 - r.t0) >= 0.5 and o > bo:
                    best, bo = j, o
            if best is not None:
                used.add(best)
            rows.append(
                dict(
                    male=r.indiv,
                    label=r.label,
                    det=best is not None,
                    pred=pred[best]['label'] if best is not None else 'missed',
                )
            )
        fa += sum(
            1
            for j, p in enumerate(pred)
            if j not in used
            and not any(min(p['t1'], r.t1) > max(p['t0'], r.t0) for r in g.itertuples())
        )
        span_total += (hi - lo) / 60
    R = pd.DataFrame(rows)
    out = dict(
        pipeline=pipe,
        model=os.path.basename(spec.rstrip('/')),
        songs=len(R),
        song_recall=float(R.det.mean()),
        false_alarms_per_min=float(fa / max(span_total, 1e-9)),
    )
    for c in ['BhBl', 'Unfinished']:
        m = R.label == c
        out[f'{c}_recall_end2end'] = float((R[m].pred == c).mean())
        out[f'{c}_pred'] = R[m].pred.value_counts().to_dict()
    # per-male BhBl recall + male-cluster bootstrap
    b = R[R.label == 'BhBl']
    per = b.groupby('male').apply(lambda d: (d.pred == 'BhBl').mean())
    out['BhBl_recall_per_male'] = per.round(2).to_dict()
    rng = np.random.default_rng(0)
    males = b.male.unique()
    bs = []
    for _ in range(1000):
        s = rng.choice(males, len(males))
        d = pd.concat([b[b.male == m] for m in s])
        bs.append((d.pred == 'BhBl').mean())
    out['BhBl_recall_CI95'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    print(json.dumps(out))
    json.dump(out, open(f'{WS}/results/e2e_{pipe}_{out["model"]}.json', 'w'), indent=1)
