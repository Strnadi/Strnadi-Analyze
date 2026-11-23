"""Index the original labeled dataset (new-data.zip) with recording-level groups.

Groups: NCS clips F00xxxx.<ext>.<seg>.wav -> F00xxxx ; BirdNET-mobile clips -> device+recording id;
aug_1_* copies are grouped with their originals (they are pre-baked augmentations).
"""

from config import WS
import soundfile as sf, glob, numpy as np, re, pandas as pd, os, sys

root = sys.argv[1] if len(sys.argv) > 1 else f'{WS}/data/orig_dataset'
out = sys.argv[2] if len(sys.argv) > 2 else f'{WS}/data/orig_index.csv'
rows = []
for f in sorted(glob.glob(root + '/*/*.wav')):
    c = f.split('/')[-2]
    n = os.path.basename(f)
    y, sr = sf.read(f, dtype='int16')
    z = y == 0
    if z.any():
        d = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
        s = np.where(d == 1)[0]
        e = np.where(d == -1)[0]
        lz = (e - s).max() / sr
    else:
        lz = 0
    nz = np.nonzero(y)[0]
    tail = (len(y) - 1 - nz[-1]) / sr if len(nz) else 4.0
    aug = n.startswith('aug_')
    base = re.sub(r'^aug_\d+_', '', n)
    lon = lat = None
    m = re.match(r'(F\d+)\.(\w+)\.(\d+)\.wav$', base)
    if m:
        src = 'ncs'
        group = m.group(1)
        seg = int(m.group(3))
    else:
        m2 = re.search(r'birdnet_mobile_(\d+)_recording_(\d+)(?:_(\d+))?\.wav$', base)
        src = 'birdnet_mobile'
        group = 'bm_' + m2.group(1) + '_' + m2.group(2)
        seg = int(m2.group(3) or 0)
        p = base.split('_')
        try:
            lon, lat = float(p[0]), float(p[1])
        except ValueError:
            pass
    rows.append(
        dict(
            path=f,
            name=n,
            base=base,
            label=c,
            src=src,
            aug=aug,
            group=group,
            seg=seg,
            max_zero_run=lz,
            tail_zero=tail,
            peak=int(np.abs(y).max()),
            lon=lon,
            lat=lat,
        )
    )
df = pd.DataFrame(rows)
df.to_csv(out, index=False)
print(pd.crosstab(df.label, [df.src, df.aug]))
print('tail zero >0.1s:')
print(pd.crosstab(df.label, [df.src, df.tail_zero > 0.1]))
print('internal zero-run>0.05s:')
print(pd.crosstab(df.label, df.max_zero_run > 0.05))
print(
    'groups:',
    df.group.nunique(),
    ' multi-label groups:',
    (df.groupby('group').label.nunique() > 1).sum(),
)
print(
    'aug whose base also present:', df[df.aug].base.isin(df[~df.aug].base).sum(), '/', df.aug.sum()
)
print(
    df[df.src == 'birdnet_mobile']
    .groupby('label')[['lon', 'lat']]
    .agg(['min', 'median', 'max'])
    .round(1)
)
