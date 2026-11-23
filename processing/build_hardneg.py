"""Hard negatives: windows where BirdNET hears another species (p>=0.5) but no Yellowhammer (<0.02), from NCS and
Vizina-train recordings; inherit the recording's fold/split from song_index.  Label 'None'."""

from config import BIRDNET_DIR, WS
import glob, numpy as np, pandas as pd

B = pd.concat([pd.read_parquet(p) for p in glob.glob(f'{WS}/birdnet24/*.parquet')])
S = pd.read_parquet(f'{WS}/data/song_index.parquet')
meta = S.groupby('npy').agg(
    rec=('rec', 'first'),
    fold=('fold', 'first'),
    split=('split', 'first'),
    source=('source', 'first'),
)
H = B[(B.yh < 0.02) & (B.other_p >= 0.5)].merge(meta, left_on='npy', right_index=True)
H = H[(H.source == 'ncs') | ((H.source == 'vizina') & (H.split == 'train'))]
H = H.sample(frac=1, random_state=0).groupby('npy').head(3)
H = H.assign(t0=(H.t - 0.5).clip(lower=0), label='None', aug=False, w=0.7, sw=1.0)
H['t1'] = H.t0 + 4.0
H['name'] = H.rec + '@hn' + H.t0.round(1).astype(str)
labels = open(f'{BIRDNET_DIR}/v2.4/tf/labels/en_us.txt').read().strip().split('\n')
H['other'] = H.other_idx.map(lambda i: labels[i].split('_')[-1] if i < len(labels) else '?')
H[
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
        'other',
        'other_p',
    ]
].to_csv(f'{WS}/data/hardneg_v1.csv', index=False)
print(len(H), H.source.value_counts().to_dict())
print(H.other.value_counts().head(12).to_dict())
