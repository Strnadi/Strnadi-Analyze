"""Run BirdNET 2.4 (TFLite, single-thread per process) over cached 32 kHz audio; store per-window YH score
plus top competing species. 3 s windows, configurable hop. Output: one parquet per worker shard.

Usage: run_birdnet_corpus.py <sources comma> <hop_s> <n_procs> <out_dir>
"""
from config import BIRDNET_DIR, WS
import os, sys, numpy as np, pandas as pd, time
from multiprocessing import Pool
from scipy.signal import resample_poly

MODEL = f'{BIRDNET_DIR}/v2.4/tf/model-fp32.tflite'
LABELS = f'{BIRDNET_DIR}/v2.4/tf/labels/en_us.txt'


def worker(args):
    shard, files, hop, out_dir = args
    out = f'{out_dir}/shard{shard:03d}.parquet'
    if os.path.exists(out): return out
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=MODEL, num_threads=1)
    B = 16
    it.resize_tensor_input(0, [B, 144000]); it.allocate_tensors()
    iin = it.get_input_details()[0]['index']; iout = it.get_output_details()[0]['index']
    species = open(LABELS).read().strip().split('\n') if os.path.exists(LABELS) else None
    yh = 2098
    rows = []
    for npy in files:
        try:
            a = np.load(npy, mmap_mode='r')
        except Exception:
            continue
        dur = len(a) / 32000
        starts = np.arange(0, max(dur - 3.0, 0) + 1e-6, hop)
        if len(starts) == 0: starts = np.array([0.0])
        for s0 in range(0, len(starts), B):
            st = starts[s0:s0 + B]
            X = np.zeros((B, 144000), np.float32)
            for j, t in enumerate(st):
                seg = np.asarray(a[int(t * 32000):int(t * 32000) + 96000], np.float32) / 32768
                seg = resample_poly(seg, 3, 2).astype(np.float32)[:144000]
                X[j, :len(seg)] = seg
            it.set_tensor(iin, X); it.invoke()
            L = it.get_tensor(iout)[:len(st)]
            P = 1 / (1 + np.exp(-np.clip(L, -20, 20)))
            top = np.argsort(-P, 1)[:, :2]
            for j, t in enumerate(st):
                o = top[j, 0] if top[j, 0] != yh else top[j, 1]
                rows.append((npy, float(t), float(P[j, yh]), int(o), float(P[j, o])))
    df = pd.DataFrame(rows, columns=['npy', 't', 'yh', 'other_idx', 'other_p'])
    df.to_parquet(out)
    return out


if __name__ == '__main__':
    sources, hop, n, out_dir = sys.argv[1].split(','), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    os.makedirs(out_dir, exist_ok=True)
    man = pd.read_csv(f'{WS}/audio32k/manifest.csv')
    man = man[man.ok & man.source.isin(sources)].sort_values('dur', ascending=False)
    files = man.npy.tolist()
    # balance shards by duration (greedy)
    nsh = n * 6
    shards = [[] for _ in range(nsh)]; load = np.zeros(nsh)
    for f, d in zip(man.npy, man.dur):
        i = int(load.argmin()); shards[i].append(f); load[i] += d
    t = time.time()
    with Pool(n) as pool:
        for i, o in enumerate(pool.imap_unordered(worker, [(i, s, hop, out_dir) for i, s in enumerate(shards)])):
            print(i, o, round(time.time() - t), flush=True)
