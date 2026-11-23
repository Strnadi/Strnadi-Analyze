"""End-to-end check of the service pipeline on a real annotated recording (needs the workspace and an exported
model): runs `serving.pipeline.SongClassifier` on ~90 s of a held-out Vizina male and prints its segments next to
the annotated songs, plus the edge cases the old runner crashed on (silence, a clip shorter than one window).

    python -m serving.export_onnx <run_dir> model.onnx
    python -m tests.check_serving model.onnx
"""
import sys
import numpy as np
import pandas as pd
from config import WS, CLASSES, SR
from serving.pipeline import SongClassifier


def main():
    clf = SongClassifier(sys.argv[1], CLASSES, threads=1)
    V = pd.read_csv(f'{WS}/data/vizina_eval.csv', dtype={'indiv': str}); V = V[V.split == 'test']
    npy, g = max(V.groupby('npy'), key=lambda kv: len(kv[1])); g = g.sort_values('t0')
    lo = max(0.0, g.t0.iloc[0] - 3); hi = lo + 90
    y = np.asarray(np.load(npy, mmap_mode='r')[int(lo * SR):int(hi * SR)], np.float32) / 32768
    out = clf.classify(y); truth = g[(g.t1 <= hi)]
    print(f'annotated songs in the excerpt: {len(truth)} ({truth.label.value_counts().to_dict()})')
    print(f'pipeline segments: {len(out["segments"])}, representant {out["representantId"]}')
    hit = 0
    for r in truth.itertuples():
        m = [s for s in out['segments'] if min(s['interval'][1], r.t1 - lo) - max(s['interval'][0], r.t0 - lo) >= 0.5 * (r.t1 - r.t0)]
        hit += bool(m); print(f'  {r.t0 - lo:6.1f}-{r.t1 - lo:6.1f} s  {r.label:10s} -> {m[0]["label"] if m else "missed"}')
    print(f'detected {hit} of {len(truth)}')
    seg = out['segments'][0] if out['segments'] else None
    assert seg is None or (abs(sum(seg['fullPredictions'].values()) - 1) < 1e-3 and set(seg) == {'interval', 'label', 'isRepresentant', 'fullPredictions'})
    assert clf.classify(np.zeros(5 * SR, np.float32)) == {'representantId': -1, 'segments': []}           # silence: no crash, empty result
    assert isinstance(clf.classify(y[:SR])['segments'], list)                                              # shorter than one window
    print('serving ok')


if __name__ == '__main__':
    main()
