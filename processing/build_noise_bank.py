"""Background-noise bank: 6 s chunks from AudioMoth files >= 10 s away from any BirdNET Yellowhammer detection.
Stored as list of float16 arrays (object npy).  Also writes AudioMoth YH detection table for later use."""
from config import SRC, WS
import glob, csv, os, numpy as np, pandas as pd
SRC = f'{SRC}/source/bhbl'
CACHE = f'{WS}/audio32k/audiomoth_bhbl'
rng = np.random.default_rng(0)
det_rows, bank = [], []
for tab in sorted(glob.glob(SRC + '/*/*.selection.table.txt')):
    rel = os.path.relpath(tab, SRC)
    wav = rel.replace('.BirdNET.selection.table.txt', '.WAV')
    npy = f'{CACHE}/{wav}.npy'
    if not os.path.exists(npy): continue
    dets = []
    for r in csv.DictReader(open(tab), delimiter='\t'):
        try:
            t0, t1, c = float(r['Begin Time (s)']), float(r['End Time (s)']), float(r['Confidence'])
        except ValueError:
            continue
        if r['Common Name'] == 'Yellowhammer':
            dets.append((t0, t1, c)); det_rows.append((npy, t0, t1, c))
    a = np.load(npy, mmap_mode='r'); dur = len(a) / 32000
    busy = np.zeros(int(dur) + 1, bool)
    for t0, t1, _ in dets:
        busy[max(0, int(t0) - 10):int(t1) + 10] = True
    free = np.where(~busy[:-7])[0]
    for s in rng.choice(free, size=min(len(free), 12), replace=False) if len(free) else []:
        x = np.asarray(a[s * 32000:(s + 6) * 32000], np.float32) / 32768
        if len(x) == 6 * 32000 and np.abs(x).max() > 1e-4:
            bank.append(x.astype(np.float16))
pd.DataFrame(det_rows, columns=['npy', 't0', 't1', 'conf']).to_csv(f'{WS}/data/audiomoth_yh_dets.csv', index=False)
arr = np.empty(len(bank), dtype=object); arr[:] = bank
np.save(f'{WS}/data/noise_bank_audiomoth.npy', arr, allow_pickle=True)
print('noise chunks', len(bank), 'YH detections', len(det_rows))
