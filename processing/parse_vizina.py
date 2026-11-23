"""Parse Vizina Raven annotation tables into a song-level table.

Each song has a 'song' row (begin/end) plus component rows (first phrase, dialect_1, dialect_2) sharing song_code.
complete = has dialect_1 and dialect_2 components -> BhBl ; otherwise Unfinished (B-only or first phrase only).
"""

from config import SRC, WS
import glob, csv, re, os, collections, sys
import pandas as pd

ROOT = f'{SRC}/source/Vizina_YH_all_rec'
rows = []
for f in glob.glob(ROOT + '/Vizina_annotation/*.txt'):
    with open(f, encoding='utf-8', errors='replace') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            r['_f'] = os.path.basename(f)
            rows.append(r)


def norm_comp(c):
    c = (c or '').strip().lower().replace(' ', '')
    if c.startswith('song'):
        return 'song'
    if c.startswith('first'):
        return 'first'
    m = re.match(r'd[ia]+l?[ae]*c?t?_?(\d)', c)
    if m:
        return 'dialect_' + m.group(1)
    return c


songs = {}
comps = collections.defaultdict(list)
for r in rows:
    comp = norm_comp(r.get('Component'))
    key = (r['_f'], (r.get('song_code') or '').strip())
    b, e = float(r['Begin Time (s)']), float(r['End Time (s)'])
    if comp == 'song':
        songs[key] = dict(
            ann=r['_f'],
            song_code=key[1],
            indiv=(r.get('Indi_code') or '').strip(),
            t0=b,
            t1=e,
            dialect_type=(r.get('dialect_type') or '').strip(),
        )
    else:
        comps[key].append(
            (
                comp,
                b,
                e,
                (r.get('Phrase_type') or '').strip(),
                (r.get('dialect_type') or '').strip(),
            )
        )
out = []
for key, s in songs.items():
    c = comps.get(key, [])
    kinds = [x[0] for x in c]
    s['has_first'] = 'first' in kinds
    s['has_d1'] = 'dialect_1' in kinds
    s['has_d2'] = 'dialect_2' in kinds
    s['phrase_type'] = next((x[3] for x in c if x[0] == 'first'), '')
    dts = [x[4] for x in c if x[4]] + ([s['dialect_type']] if s['dialect_type'] else [])
    s['dialect_any'] = ';'.join(sorted(set(d.replace(' ', '') for d in dts)))
    s['label'] = 'BhBl' if (s['has_d1'] and s['has_d2']) else 'Unfinished'
    # precise extent from component boxes (the 'song' box is a fixed 3 s detector window)
    if c:
        s['det_t0'], s['det_t1'] = s['t0'], s['t1']
        s['t0'] = min(x[1] for x in c)
        s['t1'] = max(x[2] for x in c)
        s['term_t0'] = min([x[1] for x in c if x[0].startswith('dialect')], default=float('nan'))
    out.append(s)
df = pd.DataFrame(out)
# map annotation file -> audio file (annotation names = '<audio stem>.Learning.Detector.selections.txt')
df['audio_stem'] = df.ann.str.replace(
    r'\.Learning\.Detector\.selections\.txt$', '', regex=True
).str.replace(r'\.Table\.\d+\.selections\.txt$', '', regex=True)
audio = {
    os.path.splitext(os.path.basename(p))[0]: p
    for p in glob.glob(ROOT + '/**/*.WAV', recursive=True)
}
df['audio'] = df.audio_stem.map(audio)
df['dur'] = df.t1 - df.t0
df.to_csv(sys.argv[1] if len(sys.argv) > 1 else f'{WS}/data/vizina_songs.csv', index=False)
print(
    len(df),
    'songs;',
    df.audio.notna().sum(),
    'matched to audio;',
    df.indiv.nunique(),
    'individuals;',
    df.ann.nunique(),
    'files',
)
print(df.label.value_counts().to_dict())
print(df.groupby('label').dur.describe(percentiles=[0.05, 0.5, 0.9, 0.95, 0.99]).round(2))
print('dialect_any', df.dialect_any.value_counts().head(8).to_dict())
print(df[df.audio.isna()].ann.unique()[:5])
