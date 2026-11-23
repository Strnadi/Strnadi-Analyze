"""BirdNET 3.0 embeddings for the labelled clips, straight through onnxruntime (the birdnet package's multiprocessing
session deadlocked on this box).  Model: 32 kHz, 3 s input (96000 samples) -> 1280-d embedding.
Each 4 s clip -> 3 windows at 0 / 0.5 / 1.0 s -> spatial (N,3,1280) + emb = mean.  Run in the TF venv:
  OMP_NUM_THREADS=2 python -I extract_bn30.py [threads]"""
from config import BIRDNET_DIR, WS
import sys, glob, numpy as np, pandas as pd, onnxruntime as ort
SR = 32000; W = 96000
th = int(sys.argv[1]) if len(sys.argv) > 1 else 2
so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1
sess = ort.InferenceSession(glob.glob(f'{BIRDNET_DIR}/v3.0/onnx/model-fp32-*.onnx')[0], so, providers=['CPUExecutionProvider'])
df = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
E, Sq = [], []
for i in range(0, len(df), 32):
    X = []
    for r in df.iloc[i:i + 32].itertuples():
        a = np.load(r.npy, mmap_mode='r'); y = np.asarray(a[int(r.t0 * SR):int(r.t1 * SR)], np.float32) / 32768
        if len(y) < SR * 4: y = np.pad(y, (0, SR * 4 - len(y)))
        for o in (0, SR // 2, SR): X.append(y[o:o + W])
    emb = sess.run(['embeddings'], {'input': np.stack(X)})[0].reshape(-1, 3, 1280)
    Sq.append(emb.astype(np.float16)); E.append(emb.mean(1).astype(np.float16))
    if i % 640 == 0: print(i, flush=True)
np.savez(f'{WS}/emb/orig_bn30.npz', emb=np.concatenate(E), spatial=np.concatenate(Sq), idx=df.index.values)
print('done', len(df))
