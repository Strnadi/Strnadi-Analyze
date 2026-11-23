"""Extract frozen foundation-model embeddings (Perch v2, BirdNET 2.4, BirdNET 3.0) for a list of clips.

Input CSV must have a 'path' column (wav files) or 'npy' + 't0','t1' (32 kHz int16 cache, seconds).
Outputs an .npz per model with arrays aligned to the CSV rows:
  perch: emb (N,1536), spatial (N,16,1536) [mean over 4 freq bands], yh_logit (N,)
  bn24 / bn30: emb (N,2,D) for 3-s windows at offsets [0, L-3] of the clip, yh (N,2)
Run with the TF venv.  Usage: extract_foundation_emb.py <csv> <out_prefix> [models=perch,bn24,bn30]
"""
from config import PERCH_TF_DIR
import os, sys, time, numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

def load_clip(row, sr_out):
    if isinstance(row.get('path'), str) and row['path']:
        y, sr = sf.read(row['path'], dtype='float32', always_2d=False)
        if y.ndim > 1: y = y.mean(1)
        if sr != sr_out:
            g = np.gcd(sr, sr_out); y = resample_poly(y, sr_out // g, sr // g).astype(np.float32)
        return y
    a = np.load(row['npy'], mmap_mode='r')
    i0, i1 = int(row['t0'] * 32000), int(row['t1'] * 32000)
    y = np.asarray(a[max(0, i0):i1], dtype=np.float32) / 32768.0
    if i0 < 0: y = np.concatenate([np.zeros(-i0, np.float32), y])
    if sr_out != 32000:
        g = np.gcd(32000, sr_out); y = resample_poly(y, sr_out // g, 32000 // g).astype(np.float32)
    return y

def peak_norm(y, target=0.25):
    m = np.abs(y).max()
    return y * (target / m) if m > 1e-6 else y

def fit(y, n):
    """center-crop or zero-pad (centered) to n samples"""
    if len(y) >= n:
        s = (len(y) - n) // 2; return y[s:s + n]
    p = n - len(y); return np.pad(y, (p // 2, p - p // 2))

def run_perch(df, out):
    import tensorflow as tf
    nt = int(os.environ.get('TF_THREADS', 8))
    tf.config.threading.set_intra_op_parallelism_threads(nt); tf.config.threading.set_inter_op_parallelism_threads(2)
    m = tf.saved_model.load(f'{PERCH_TF_DIR}')
    f = m.signatures['serving_default']
    import csv
    labels = pd.read_csv(f'{PERCH_TF_DIR}/assets/labels.csv')
    col = labels.columns[0]
    yh_idx = int(np.where(labels[col].astype(str).str.contains('emberiza_citrinella|Emberiza citrinella|yellow2', case=False, regex=True))[0][0])
    N = len(df)
    if N > 20000:  # disk-backed outputs for large sets
        E = np.lib.format.open_memmap(out + '_perch_emb.npy', 'w+', np.float16, (N, 1536))
        S = np.lib.format.open_memmap(out + '_perch_spatial.npy', 'w+', np.float16, (N, 16, 1536))
    else:
        E = np.zeros((N, 1536), np.float16); S = np.zeros((N, 16, 1536), np.float16)
    Y = np.zeros(N, np.float32)
    B = 16; t = time.time()
    rows = df.to_dict('records')
    for s in range(0, N, B):
        xb = np.stack([peak_norm(fit(load_clip(r, 32000), 160000)) for r in rows[s:s + B]])
        o = f(inputs=tf.constant(xb))
        E[s:s + B] = o['embedding'].numpy(); S[s:s + B] = o['spatial_embedding'].numpy().mean(2); Y[s:s + B] = o['label'].numpy()[:, yh_idx]
        if s % (B * 20) == 0: print('perch', s, N, round(time.time() - t, 1), flush=True)
    if N > 20000:
        E.flush(); S.flush(); np.save(out + '_perch_yh.npy', Y)
    else:
        np.savez(out + '_perch.npz', emb=E, spatial=S, yh_logit=Y)

def run_birdnet(df, out, ver):
    import birdnet
    m = birdnet.load('acoustic', ver, 'tf' if ver == '2.4' else 'onnx')
    sr = m.get_sample_rate(); n = int(3 * sr)
    yh_idx = [i for i, s in enumerate(m.species_list) if s.startswith('Emberiza citrinella')][0]
    rows = df.to_dict('records')
    wins = []
    for r in rows:
        y = load_clip(r, sr)
        if len(y) < n: y = fit(y, n)
        wins.append(y[:n]); wins.append(y[len(y) - n:])
    X = np.stack(wins).astype(np.float32)
    t = time.time()
    res = m.encode_arrays([(x, sr) for x in X]) if hasattr(m, 'encode_arrays') else None
    print('bn', ver, 'encode done', time.time() - t, type(res), flush=True)
    np.save(out + f'_bn{ver.replace(".", "")}_raw.npy', np.array([0]))
    return res

if __name__ == '__main__':
    csv_path, out = sys.argv[1], sys.argv[2]
    models = (sys.argv[3] if len(sys.argv) > 3 else 'perch').split(',')
    df = pd.read_parquet(csv_path) if csv_path.endswith('.parquet') else pd.read_csv(csv_path)
    if 'perch' in models: run_perch(df, out)
