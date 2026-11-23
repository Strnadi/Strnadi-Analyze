"""Turn the song index into an 'extra' training table for the trainer (partial labels + None windows; song_t0/song_t1 = song span for frame targets).

Window span per song: [onset-0.6, onset+3.8] (4.4 s); the trainer draws a random 4 s crop inside it.
Rows of NCS recordings inherit the CV fold of their recording group, so weak windows of validation recordings are
never trained on.  Vizina held-out individuals ('test') are excluded entirely.
Usage: build_weak_set.py <out_csv> [min_yh] [weak_frac]
"""

from config import SRC, WS
import sys, numpy as np, pandas as pd

out = sys.argv[1]
min_yh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
weak_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
S = pd.read_parquet(f'{WS}/data/song_index.parquet')
N = pd.read_parquet(f'{WS}/data/none_index.parquet')
S = S[
    (S.weak != '') & (S.split == 'train') & (S.yh >= min_yh) & ((S.seg_t1 - S.seg_t0) <= 3.8)
].copy()
S['t0'] = S.seg_t0 - 0.6
S['t1'] = S.t0 + 4.4
S['song_t0'] = S.seg_t0
S['song_t1'] = S.seg_t1
S = S[S.t0 >= 0]
S['label'] = S.weak.str.replace(
    'BE|BhBl|', 'BE|BlBh|', regex=False
)  # BD -> mostly BlBh after 2023 re-evaluation (review V8)
# expert re-evaluated dialects (Diblikova et al. 2023, Table S1) override file labels for those recordings
T = pd.read_excel(f'{SRC}/external/diblikova2023/Supplementary_Table_S1.xlsx')
T['F'] = T.filename.str.extract(r'^(F\d+)')[0]
newd = {
    f: {'BE': 'BE', 'BlBh': 'BlBh', 'transitional': 'BE|BlBh'}[d]
    for f, d in zip(T.F, T.new_dialect)
    if isinstance(f, str)
}
allS = pd.read_parquet(f'{WS}/data/song_index.parquet')
exp = allS[
    (allS.source == 'ncs')
    & allS.rec.str.replace('ncs:', '').isin(newd)
    & (allS.split == 'train')
    & (allS.yh >= min_yh)
    & ((allS.seg_t1 - allS.seg_t0) <= 3.8)
].copy()
exp['t0'] = exp.seg_t0 - 0.6
exp['t1'] = exp.t0 + 4.4
exp['song_t0'] = exp.seg_t0
exp['song_t1'] = exp.seg_t1
exp = exp[exp.t0 >= 0]
exp['label'] = exp.rec.str.replace('ncs:', '').map(newd) + '|Unfinished|None'
S = pd.concat([S[~S.rec.isin(exp.rec)], exp], ignore_index=True)
# weak-only NCS recordings get a hashed fold (instead of -1 = always train) -> every weak recording is evaluable out-of-fold
import zlib

m = (S.source == 'ncs') & (S.fold < 0)
if m.any():
    S.loc[m, 'fold'] = S.loc[m, 'rec'].map(lambda r: zlib.crc32(r.encode()) % 5).astype(int)
N = N[N.split == 'train'].copy()
N['t0'] = N.seg_t0
N['t1'] = N.seg_t1
N['label'] = 'None'
E = pd.concat([S, N], ignore_index=True)
E['aug'] = False
E['name'] = E.rec + '@' + E.t0.round(2).astype(str)
E['w'] = np.where(E.label == 'None', 0.5, 1.0)
# cap songs per recording (long AudioMoth/Vizina files would dominate) and sampling weight so weak ~ weak_frac of draws
E['n_in_rec'] = E.groupby('rec').npy.transform('size')
E['sw'] = 1.0 / np.sqrt(E.n_in_rec)
n_strong = 3744
E['sw'] *= (weak_frac / (1 - weak_frac)) * n_strong / E.sw.sum()
E[
    [
        'npy',
        't0',
        't1',
        'label',
        'fold',
        'aug',
        'name',
        'w',
        'sw',
        'source',
        'rec',
        'song_t0',
        'song_t1',
    ]
].to_csv(out, index=False)
print(E.groupby(['source', 'label']).size().unstack(fill_value=0).T)
print('rows', len(E), 'recs', E.rec.nunique())
