"""Does a model learn "passive-recorder acoustics => BhBl"?

Every passive-recorder training song (AudioMoth sample, JCU) is constrained to {BhBl, Unfinished, None}, because
those sites only have BhBl. A model can satisfy that by keying on far-field / AudioMoth acoustics instead of the
song ending, and the JCU "wrong-dialect rate" metric rewards exactly that. It would then mislabel the Italian
passive recordings, where the dialects are unknown.

Test: take complete, correctly labelled songs of the OTHER dialects (BC, BE, BlBh, XB) from citizen recordings,
out of fold; make them sound like a passive recording (far-field filter + reverberation + real AudioMoth noise at
measured SNRs); count how often the prediction moves to BhBl. Compare a model trained without passive data
(e.g. base_v3) with one trained with it (R6): a clearly higher BhBl rate is the shortcut.
The same degraded clips also give the far-field robustness of each model (how many stay correct).

Usage: python -m evaluation.dialect_prior_shift <run> [<run> ...] [--noise data/noise_bank_clean.npy] [--n 400]
"""

import argparse
import glob
import json
import os
import numpy as np
import torch
from config import WS, CLASSES, C2I
from augment.far_field import far_field, load_profiles
from augment.noise_mix import mix_noise, draw_chunk
from training.checkpoint import load_model
from training.data import read_table, load_seg
from config import workspace_path
from training.inference import peak_normalize

N = 128000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--noise', default='data/noise_bank_audiomoth.npy')
    ap.add_argument('--table', default='data/orig_index.csv')
    ap.add_argument('--n', type=int, default=400)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('THREADS', 2)))
    data = read_table(a.table)
    aug = data['aug'] if 'aug' in data else False
    pool = data[~aug & data.label.isin(['BC', 'BE', 'BlBh', 'XB']) & (data.src == 'ncs')]
    pool = pool.sample(min(a.n, len(pool)), random_state=0)
    bank = list(np.load(workspace_path(a.noise), allow_pickle=True))
    profiles = load_profiles(f'{WS}/data/farfield_stats.csv')
    if not profiles:
        raise SystemExit('data/farfield_stats.csv is missing')
    np.random.seed(0)
    clean = [load_seg(r.npy, r.t0, r.t0 + 4, N) for r in pool.itertuples()]
    far = []
    for y in clean:
        z, snr = far_field(y, profiles[np.random.randint(len(profiles))])
        far.append(mix_noise(z, draw_chunk(bank, N), snr, 'ending'))
    Xc, Xf = peak_normalize(np.stack(clean)), peak_normalize(np.stack(far))
    y = pool.label.map(C2I).values
    for run in a.runs:
        pc, pf = np.zeros((len(pool), 7)), np.zeros((len(pool), 7))
        for pt in sorted(glob.glob(f'{WS}/runs/{run}/fold*/model.pt')):
            k = int(os.path.basename(os.path.dirname(pt)).replace('fold', ''))
            m = (pool.fold == k).values
            if not m.any():
                continue
            model = load_model(pt)
            with torch.no_grad():
                pc[m] = torch.softmax(model(torch.from_numpy(Xc[m]))['logits'], 1).numpy()
                pf[m] = torch.softmax(model(torch.from_numpy(Xf[m]))['logits'], 1).numpy()
        done = pc.sum(1) > 0
        ok = done & (pc.argmax(1) == y)  # songs the model gets right when clean
        res = dict(
            run=run,
            n_correct_clean=int(ok.sum()),
            to_BhBl_after_farfield=float((pf[ok].argmax(1) == C2I['BhBl']).mean()),
            to_BhBl_clean=float((pc[done].argmax(1) == C2I['BhBl']).mean()),
            still_correct=float((pf[ok].argmax(1) == y[ok]).mean()),
            to_Unfinished=float((pf[ok].argmax(1) == C2I['Unfinished']).mean()),
            pred_dist_farfield={
                CLASSES[c]: float((pf[ok].argmax(1) == c).mean()) for c in range(7)
            },
        )
        print(json.dumps(res))
        json.dump(res, open(f'{WS}/results/prior_shift_{run}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
