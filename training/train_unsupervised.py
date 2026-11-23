"""Label-free training of the legacy unsupervised models on detected songs:

  --objective reconstruction   autoencoders (conv_autoencoder, mobile_autoencoder): MSE between input and output
  --objective nt_xent          contrastive embedding models (densenet_crnn_embed): two augmented views per song

Songs come from the song index (training split only: no held-out Vizina males, no NCS audio, same corpus rule as
`ssl_pretrain.py`). Views for the contrastive objective are made at waveform level (speed, device EQ, real noise),
not just by SpecAugment inside the model as in the Keras script.
Writes $STRNADI_WS/ssl/<name>/model.pt (+ log.json). For StrnadNet encoders use `training/ssl_pretrain.py`.

Usage: python -m training.train_unsupervised --name <name> --model conv_autoencoder --objective reconstruction
"""
import argparse
import json
import math
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from config import WS
from model import build_model, count_params
from layers.spec_augment import SpecAugment
from augment.pipeline import WaveAugment, peak_normalize
from losses.nt_xent import nt_xent
from training.data import load_seg

N = 128000


class Songs(torch.utils.data.Dataset):
    def __init__(self, songs, augment, two_views):
        self.s, self.aug, self.two = songs.reset_index(drop=True), augment, two_views

    def __len__(self):
        return 10 ** 9

    def __getitem__(self, _):
        r = self.s.iloc[np.random.randint(len(self.s))]; t0 = r.seg_t0 - np.random.uniform(0.2, 0.6)
        y = load_seg(r.npy, t0, t0 + 4, N)
        if self.two:
            return torch.from_numpy(self.aug(y.copy())), torch.from_numpy(self.aug(y.copy()))
        return torch.from_numpy(peak_normalize(y)), 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--objective', choices=['reconstruction', 'nt_xent'], required=True); ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--bs', type=int, default=32); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2); ap.add_argument('--sources', default='vizina,audiomoth_bhbl,birdnet_yh')
    ap.add_argument('--noise', default='data/noise_bank_clean.npy'); ap.add_argument('--pretrained', type=int, default=1)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    S = pd.read_parquet(f'{WS}/data/song_index.parquet')
    S = S[S.source.isin(a.sources.split(',')) & ((S.yh >= 0.3) | S.yh.isna()) & ((S.seg_t1 - S.seg_t0) <= 3.8) & (S.split == 'train') & (S.seg_t0 >= 0.6)]
    noise_path = a.noise if os.path.isabs(a.noise) else f'{WS}/{a.noise}'
    noise = list(np.load(noise_path, allow_pickle=True)) if os.path.exists(noise_path) else None
    augment = WaveAugment(dict(speed=0.02, p_noise=0.8, snr=(0.0, 20.0), snr_mode='band', p_eq=0.7), noise)
    two = a.objective == 'nt_xent'
    kw = dict(pretrained=bool(a.pretrained)) if a.model == 'densenet_crnn_embed' else {}
    model = build_model(a.model, **kw); print(a.model, 'params', count_params(model), 'songs', len(S), flush=True)
    spec_aug = SpecAugment(fill=getattr(model, 'spec_fill', 'zero')).train()
    dl = iter(torch.utils.data.DataLoader(Songs(S, augment, two), batch_size=a.bs, num_workers=a.workers))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 300) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    out_dir = f'{WS}/ssl/{a.name}'; os.makedirs(out_dir, exist_ok=True); log, t0 = [], time.time(); model.train()
    for step in range(a.steps):
        x1, x2 = next(dl)
        if two:
            z1 = model(spec_aug(model.frontend(x1)))['emb']; z2 = model(spec_aug(model.frontend(x2)))['emb']
            loss = nt_xent(z1, z2, 0.1)
        else:
            out = model(x1); loss = F.mse_loss(out['recon'], out['target'])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0); opt.step(); sched.step()
        if step % 50 == 0:
            log.append(dict(step=step, loss=loss.item(), t=time.time() - t0)); print(log[-1], flush=True)
        if (step + 1) % 1000 == 0 or step == a.steps - 1:
            torch.save(dict(state=model.state_dict(), model=a.model, model_kw=kw, args=vars(a), step=step + 1), f'{out_dir}/model.pt')
            json.dump(log, open(f'{out_dir}/log.json', 'w'))


if __name__ == '__main__':
    main()
