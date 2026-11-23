"""Decode every audio source into a uniform cache: 32 kHz mono int16 .npy (memory-mappable).

Writes a manifest CSV with source, relative path, duration, original sample-rate/channels, peak.
Usage: python decode_all.py <out_root> <n_procs>
"""

from config import SRC
import os, sys, glob, subprocess, json, numpy as np, pandas as pd
from multiprocessing import Pool
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
SR = 32000
SRC = f'{SRC}/source'
AUDIO_EXT = {
    '.wav',
    '.mp3',
    '.wma',
    '.m4a',
    '.mp4',
    '.mov',
    '.avi',
    '.wmv',
    '.3ga',
    '.amr',
    '.3gp',
    '.3gpp',
    '.flac',
    '.mts',
    '.m2ts',
    '.ogg',
}


def list_jobs():
    jobs = []
    for source, sub in [
        ('ncs', 'ncs_nahravky'),
        ('vizina', 'Vizina_YH_all_rec'),
        ('audiomoth_bhbl', 'bhbl'),
        ('birdnet_yh', 'birdnet_yh'),
    ]:
        for p in glob.glob(f'{SRC}/{sub}/**/*', recursive=True):
            if os.path.isfile(p) and os.path.splitext(p)[1].lower() in AUDIO_EXT:
                jobs.append((source, p, os.path.relpath(p, f'{SRC}/{sub}')))
    return jobs


def decode(job):
    source, path, rel = job
    out = os.path.join(OUT, source, rel + '.npy')
    if os.path.exists(out):
        try:
            a = np.load(out, mmap_mode='r')
            return dict(
                source=source, rel=rel, path=path, npy=out, dur=len(a) / SR, ok=True, err=''
            )
        except Exception:
            pass
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [
        FF,
        '-nostdin',
        '-v',
        'error',
        '-i',
        path,
        '-vn',
        '-ac',
        '1',
        '-ar',
        str(SR),
        '-f',
        's16le',
        '-acodec',
        'pcm_s16le',
        '-',
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
        a = np.frombuffer(r.stdout, dtype=np.int16)
        if len(a) < SR * 0.5:
            return dict(
                source=source,
                rel=rel,
                path=path,
                npy='',
                dur=len(a) / SR,
                ok=False,
                err=r.stderr.decode(errors='ignore')[-300:],
            )
        np.save(out + '.tmp.npy', a)
        os.replace(out + '.tmp.npy', out)
        return dict(
            source=source,
            rel=rel,
            path=path,
            npy=out,
            dur=len(a) / SR,
            ok=True,
            err='',
            peak=int(np.abs(a).max()),
        )
    except Exception as e:
        return dict(source=source, rel=rel, path=path, npy='', dur=0, ok=False, err=str(e)[-300:])


if __name__ == '__main__':
    OUT = sys.argv[1]
    n = int(sys.argv[2])
    jobs = list_jobs()
    # big files first for better load balancing
    jobs.sort(key=lambda j: -os.path.getsize(j[1]))
    print('jobs', len(jobs), flush=True)
    res = []
    with Pool(n, initializer=globals().update, initargs=({'OUT': OUT},)) as pool:
        for i, r in enumerate(pool.imap_unordered(decode, jobs, chunksize=4)):
            res.append(r)
            if i % 2000 == 0:
                print(i, r['source'], r['rel'][:60], r['ok'], flush=True)
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(OUT, 'manifest.csv'), index=False)
    print(
        df.groupby('source').agg(
            n=('ok', 'size'), ok=('ok', 'sum'), hours=('dur', lambda x: x.sum() / 3600)
        )
    )
