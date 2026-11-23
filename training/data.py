"""Tables -> tensors.

A table row is one window of one cached recording: `npy` (32 kHz int16 mono array on disk), `t0`, `t1` (seconds),
`label`, plus optional `fold`, `grp`, `rec`, `source`, `w` (loss weight), `sw` (sampling weight),
`song_t0` / `song_t1` (song span in the file, for frame targets), `p_<class>` (soft labels).

Label forms:  'BC'                    hard label
              'BC|Unfinished|None'    candidate set (partial label)
              'soft'                  soft label in the p_<class> columns (pseudo-labels)
"""
import json
import os
import numpy as np
import pandas as pd
import torch
from config import CLASSES, C2I, SR
from .defaults import ws

FRAME_RATE = 50          # frame targets are produced at 20 ms; losses resample them to the model's frame rate
ELEMENTS = ['bg', 'phrase', 'B', 'C', 'D', 'E', 'X']                                   # = model/element_tagger.py
MEASUREMENTS = ['f1', 'f2', 'f3', 'df', 'bw', 't2', 'has_first', 'has_second']          # = layers/measurement_head.py; columns m_<name>


def read_table(path, **kw):
    return pd.read_csv(ws(path), keep_default_na=False, na_values=[''], **kw)


def load_seg(npy, t0, t1, n):
    """n samples starting at t0 (shifted to stay inside the file). Short files are padded with low-level noise:
    never zeros, never reflection (reversed audio turns BhBl into a BlBh-like pattern)."""
    a = np.load(npy, mmap_mode='r')
    i0 = max(0, min(int(round(t0 * SR)), len(a) - n))
    y = np.asarray(a[i0:i0 + n], dtype=np.float32) / 32768.0
    if len(y) < n:
        sd = float(np.std(y)) * 0.05 + 1e-5 if len(y) else 1e-4
        y = np.concatenate([y, np.random.default_rng(len(y)).normal(0, sd, n - len(y)).astype(np.float32)])
    return y


def encode_label(row, pll_eps=0.0):
    """-> (target (C,), kind)   kind: 0 hard, 1 candidate set, 2 soft."""
    lab = row['label']; target = np.zeros(len(CLASSES), np.float32)
    if lab == 'soft':
        return np.array([row[f'p_{c}'] for c in CLASSES], np.float32), 2
    if '|' in lab:
        if str(row.get('source', '')) == 'ncs':
            target[:] = pll_eps           # escape mass: a citizen recording's file label can be wrong
        for c in lab.split('|'):
            target[C2I[c]] = 1
        return target, 1
    target[C2I[lab]] = 1
    return target, 0


class EmbeddingStore:
    """name -> cached Perch 2.0 embedding(s) of the unaugmented clip (processing/extract_foundation_emb.py)."""

    def __init__(self, spatial=False):
        from config import WS
        self.stores, self.idx, self.spatial = [], {}, spatial
        z = np.load(f'{WS}/emb/orig_perch.npz'); orig = read_table('data/orig_index.csv')
        self.stores.append(z['spatial'] if spatial else z['emb']); self.idx.update({n: (0, i) for i, n in enumerate(orig.name)})
        rows_emb = f'{WS}/emb/rows_perch_{"spatial" if spatial else "emb"}.npy'
        if os.path.exists(rows_emb) and os.path.exists(f'{WS}/emb/rows_perch_yh.npy'):
            rows = pd.read_parquet(f'{WS}/data/perch_targets_rows.parquet')
            self.stores.append(np.load(rows_emb, mmap_mode='r')); self.idx.update({n: (1, i) for i, n in enumerate(rows.name)})

    def get(self, name):
        if name in self.idx:
            b, i = self.idx[name]
            return np.asarray(self.stores[b][i], np.float32), 1.0
        return np.zeros(self.stores[0].shape[1:], np.float32), 0.0


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, df, win, train, cfg, augment=None, teacher_store=None, input_store=None):
        self.df = df.reset_index(drop=True); self.n = int(win * SR); self.win = win; self.train = train
        self.cfg, self.augment, self.teacher, self.inputs = cfg, augment, teacher_store, input_store

    def __len__(self):
        return len(self.df)

    def _start(self, r):
        slack = (r.t1 - r.t0) - self.win
        if slack <= 0.05:
            return r.t0
        if self.train:
            return r.t0 + np.random.uniform(0, slack)
        # research runs evaluated [t0, t0 + win], which cuts the end of songs longer than 3.4 s (audit); the centred
        # window treats long songs the way the training crops do on average.
        return r.t0 + (slack / 2 if self.cfg.get('eval_crop', 'center') == 'center' else 0.0)

    def _frame_target(self, r, t0, label):
        T = int(self.win * FRAME_RATE)
        if label == 'None':
            return np.zeros(T, np.float32)
        s0, s1 = r.get('song_t0', np.nan), r.get('song_t1', np.nan)
        if pd.isna(s0) or pd.isna(s1):
            return np.full(T, -1.0, np.float32)
        ft = np.zeros(T, np.float32)
        ft[max(0, int((s0 - t0) * FRAME_RATE)):max(0, min(T, int(np.ceil((s1 - t0) * FRAME_RATE))))] = 1
        return ft

    def _element_target(self, r, t0):
        """per-frame element tags from an `elements` column: JSON [[name, start_s, end_s], ...] in file time;
        frames outside every span are background (0); rows without the column are all -1 (ignored)."""
        T = int(self.win * FRAME_RATE); spans = r.get('elements', None)
        if not isinstance(spans, str) or not spans:
            return np.full(T, -1, np.int64)
        tags = np.zeros(T, np.int64)
        for name, s0, s1 in json.loads(spans):
            tags[max(0, int((s0 - t0) * FRAME_RATE)):max(0, min(T, int(np.ceil((s1 - t0) * FRAME_RATE))))] = ELEMENTS.index(name)
        return tags

    def __getitem__(self, i):
        r = self.df.iloc[i]
        target, kind = encode_label(r, self.cfg.get('pll_eps', 0.0))
        item = {'target': torch.from_numpy(target), 'kind': kind, 'w': float(r.get('w', 1.0))}
        if self.inputs is not None:                                  # cached embeddings instead of audio
            emb, found = self.inputs.get(r['name'])
            if not found:
                raise KeyError(f"no cached embedding for {r['name']}: run processing/extract_foundation_emb.py on this table first")
            item['wave'] = torch.from_numpy(emb)
            return item
        t0 = self._start(r)
        y = load_seg(r.npy, t0, t0 + self.win, self.n)
        if self.train and self.augment is not None:
            y = self.augment(y)
        else:
            pk = np.abs(y).max(); y = y / pk * 0.5 if pk > 1e-6 else y
        item['wave'] = torch.from_numpy(np.ascontiguousarray(y, np.float32))
        item['frames'] = torch.from_numpy(self._frame_target(r, t0, r['label']))
        if self.cfg.get('w_elements', 0) > 0:
            item['elements'] = torch.from_numpy(self._element_target(r, t0))
        if self.cfg.get('w_measure', 0) > 0:
            item['measure'] = torch.tensor([float(r.get(f'm_{m}', np.nan)) for m in MEASUREMENTS], dtype=torch.float32)
        if self.teacher is not None:
            e, m = self.teacher.get(r['name'])
            item['teacher_emb'], item['teacher_mask'] = torch.from_numpy(e), m
        return item


class UnlabeledDataset(torch.utils.data.Dataset):
    """FixMatch views of an unlabelled song: weak = crop + peak normalisation; strong = full augmentation + noise."""

    def __init__(self, df, win, augment):
        self.df = df.reset_index(drop=True); self.n = int(win * SR); self.win = win; self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        t0 = r.t0 + np.random.uniform(0, max(0.0, r.t1 - r.t0 - self.win))
        y = load_seg(r.npy, t0, t0 + self.win, self.n)
        weak = y / max(np.abs(y).max(), 1e-6) * 0.5
        return torch.from_numpy(weak.astype(np.float32)), torch.from_numpy(self.augment(y, force_noise=True))


class BagDataset(torch.utils.data.Dataset):
    """Bags of k songs from one recording with a recording-level dialect, for `losses/bag_complete.py`."""

    def __init__(self, df, win, augment, k=6, sources=('ncs',)):
        from config import DIALECTS
        df = df[df.source.isin(sources) & df.label.str.contains('|', regex=False)].copy()
        df['dialect'] = df.label.map(lambda s: [c for c in s.split('|') if c in DIALECTS])
        df = df[df.dialect.map(len) == 1]
        self.groups = [g for _, g in df.groupby('rec') if len(g) >= 2]
        self.k, self.n, self.win, self.augment, self.dialects = k, int(win * SR), win, augment, DIALECTS

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        g = self.groups[i]; rows = g.sample(self.k, replace=len(g) < self.k)
        waves = []
        for r in rows.itertuples():
            t0 = r.t0 + np.random.uniform(0, max(0.0, r.t1 - r.t0 - self.win))
            waves.append(self.augment(load_seg(r.npy, t0, t0 + self.win, self.n)))
        return torch.from_numpy(np.stack(waves)), self.dialects.index(rows.iloc[0].dialect[0])
