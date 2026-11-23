"""Batch pipeline for the 23 TB JCU AudioMoth share: download -> detect Yellowhammer songs -> keep song windows -> delete raw.

Selection (batch spec): every <day_step>-th recording day per device, the first <n_dawn> and last <n_dusk> files of that day
(data owner's advice: YH sing most at dawn/dusk).
Detection: tonal-SNR segmenter proposes events (cheap), BirdNET 2.4 scores only the 3 s windows covering events.
Kept per raw file (compact store, 32 kHz int16): every event with BirdNET YH >= 0.2 as a window [onset-1.5, onset+4.5]
plus up to 3 'None' windows (BirdNET YH < 0.02, far from events).  Rows go to a per-file parquet:
  npy (compact store), seg_t0/seg_t1 (song position inside the store), yh, snr, kind (song/none), src_file, src_t
Usage: jcu_pipeline.py <batch_name> <day_step> <n_dawn> <n_dusk> <n_download_threads> <n_proc>
"""

from config import BIRDNET_DIR, SRC, WS
import os, sys, time, glob, queue, threading, numpy as np, pandas as pd, soundfile as sf
from multiprocessing import Pool
from scipy.signal import resample_poly
from processing.segment import segment_songs

RAW = f'{SRC}/jcu_audiomoth/raw'
STORE = f'{WS}/audio32k/jcu'
IDX = f'{WS}/data/jcu_index'
MODEL = f'{BIRDNET_DIR}/v2.4/tf/model-fp32.tflite'
SR = 32000
_it = None


def birdnet(y32, starts):
    global _it
    if _it is None:
        from ai_edge_litert.interpreter import Interpreter

        _it = Interpreter(model_path=MODEL, num_threads=1)
        _it.resize_tensor_input(0, [16, 144000])
        _it.allocate_tensors()
    out = []
    for s0 in range(0, len(starts), 16):
        st = starts[s0 : s0 + 16]
        X = np.zeros((16, 144000), np.float32)
        for j, t in enumerate(st):
            seg = y32[int(t * SR) : int(t * SR) + 3 * SR]
            seg = resample_poly(seg, 3, 2).astype(np.float32)[:144000]
            X[j, : len(seg)] = seg
        _it.set_tensor(_it.get_input_details()[0]['index'], X)
        _it.invoke()
        L = _it.get_tensor(_it.get_output_details()[0]['index'])[: len(st), 2098]
        out += list(1 / (1 + np.exp(-np.clip(L, -20, 20))))
    return np.array(out)


def process(raw_path):
    dev = os.path.basename(os.path.dirname(raw_path))
    name = os.path.basename(raw_path)
    out_npy = f'{STORE}/{dev}/{name}.songs.npy'
    out_pq = f'{IDX}/{dev}__{name}.parquet'
    if os.path.exists(out_pq):
        os.remove(raw_path)
        return out_pq
    try:
        y, sr = sf.read(raw_path, dtype='float32')
        if y.ndim > 1:
            y = y.mean(1)
        y = (
            resample_poly(y, 2, 3).astype(np.float32)
            if sr == 48000
            else resample_poly(y, SR, sr).astype(np.float32)
        )
    except Exception as e:
        print('read fail', raw_path, e, flush=True)
        return None
    dur = len(y) / SR
    segs, _ = segment_songs(y, hi=10, lo=4, max_gap=0.3, min_dur=0.6, max_dur=6.0)
    # BirdNET: 3 windows per event at onset -1.0/-0.5/0.0 s, max (bn_hop_check2: recall@0.3 0.74 -> 0.82, FP 0.05 -> 0.07;
    # the single onset-0.5 window used for batch 1 lost ~11 % of songs at the 0.2 store threshold); windows on a 0.5 s grid, deduped
    OFFS = (-1.0, -0.5, 0.0)
    starts = sorted(
        {round(max(0.0, min(s0 + o, dur - 3)) * 2) / 2 for s0, s1, _ in segs for o in OFFS}
    )
    sc = dict(zip(starts, birdnet(y, starts))) if starts else {}
    keep, pieces, rows, pos = [], [], [], 0
    for s0, s1, snr in segs:
        yh = max(float(sc.get(round(max(0.0, min(s0 + o, dur - 3)) * 2) / 2, 0.0)) for o in OFFS)
        if yh < 0.2 or s1 - s0 > 4.5:
            continue
        a, b = int((s0 - 1.5) * SR), int((s0 + 4.5) * SR)
        if a < 0 or b > len(y):
            continue
        pieces.append(y[a:b])
        rows.append(
            dict(
                kind='song',
                seg_t0=pos / SR + 1.5,
                seg_t1=pos / SR + 1.5 + (s1 - s0),
                yh=yh,
                snr=snr,
                src_t=s0,
            )
        )
        pos += b - a
    # None windows
    busy = np.zeros(int(dur) + 2, bool)
    for s0, s1, _ in segs:
        busy[max(0, int(s0) - 5) : int(s1) + 5] = True
    free = [t for t in range(0, int(dur) - 6, 7) if not busy[t : t + 6].any()]
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    cand = list(rng.choice(free, size=min(len(free), 6), replace=False)) if free else []
    if cand:
        ys = birdnet(y, [float(t) + 1.0 for t in cand])
        for t, p in zip(cand, ys):
            if p < 0.02 and len([r for r in rows if r['kind'] == 'none']) < 3:
                pieces.append(y[int(t * SR) : int((t + 6) * SR)])
                rows.append(
                    dict(
                        kind='none',
                        seg_t0=pos / SR + 1.0,
                        seg_t1=pos / SR + 5.0,
                        yh=float(p),
                        snr=0.0,
                        src_t=float(t),
                    )
                )
                pos += 6 * SR
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    os.makedirs(IDX, exist_ok=True)
    if pieces:
        np.save(out_npy, np.clip(np.concatenate(pieces) * 32767, -32768, 32767).astype(np.int16))
    df = pd.DataFrame(rows)
    df['npy'] = out_npy
    df['src_file'] = f'{dev}/{name}'
    df['dev'] = dev
    df.to_parquet(out_pq)
    os.remove(raw_path)
    return out_pq


def select(day_step, n_dawn, n_dusk):
    m = pd.read_csv(f'{SRC}/jcu_audiomoth/manifest.csv', dtype={'file_id': str})
    m = m[m.name.str.upper().str.endswith('.WAV')].copy()
    m['dev'] = m.path.str.split('/').str[1]
    m['dt'] = pd.to_datetime(
        m.name.str.extract(r'_(\d{8}_\d{6})')[0], format='%Y%m%d_%H%M%S', errors='coerce'
    )
    m = m[m.dt.dt.year == 2025]
    m['day'] = m.dt.dt.normalize()
    sel = []
    for dev, g in m.groupby('dev'):
        days = sorted(g.day.unique())[::day_step]
        for d in days:
            gd = g[g.day == d].sort_values('dt')
            sel.append(pd.concat([gd.head(n_dawn), gd.tail(n_dusk)]).drop_duplicates('file_id'))
    return pd.concat(sel)


def main():
    batch, day_step, n_dawn, n_dusk, n_dl, n_proc = (
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
        int(sys.argv[6]),
    )
    from jcu_nas import Share

    sel = select(day_step, n_dawn, n_dusk)
    done = {os.path.basename(p)[:-8] for p in glob.glob(f'{IDX}/*.parquet')}
    sel = sel[~(sel.dev + '__' + sel.name).isin(done)]
    print(batch, 'files to fetch', len(sel), 'GB', round(sel['size'].sum() / 1e9), flush=True)
    q = queue.Queue(maxsize=n_proc * 2)
    todo = list(sel.itertuples())
    lock = threading.Lock()

    def dl_worker():
        s = Share()
        while True:
            with lock:
                if not todo:
                    return
                r = todo.pop(0)
            out = f'{RAW}/{r.dev}/{r.name}'
            os.makedirs(os.path.dirname(out), exist_ok=True)
            for attempt in range(4):
                try:
                    s.download(r.file_id, out)
                    q.put(out)
                    break
                except Exception as e:
                    print('dl retry', r.name, e, flush=True)
                    time.sleep(20 * (attempt + 1))
                    s = Share()

    ths = [threading.Thread(target=dl_worker, daemon=True) for _ in range(n_dl)]
    [t.start() for t in ths]
    t0 = time.time()
    n = 0
    with Pool(n_proc) as pool:
        pending = []
        while any(t.is_alive() for t in ths) or not q.empty() or pending:
            try:
                p = q.get(timeout=5)
                pending.append(pool.apply_async(process, (p,)))
            except queue.Empty:
                pass
            for a in [a for a in pending if a.ready()]:
                pending.remove(a)
                n += 1
                if n % 20 == 0:
                    print(
                        batch,
                        'processed',
                        n,
                        'of',
                        len(sel),
                        round(time.time() - t0),
                        's',
                        flush=True,
                    )
    print(batch, 'done', n, round(time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
