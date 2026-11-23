"""Label-free pretraining of the StrnadNet encoder on all detected Yellowhammer songs.

Objectives (weights configurable):
  supcon : same-recording / same-individual songs are positives (a male sings one dialect), otherwise two augmented
           views of the same song (SimCLR fallback).  View-specific noise/EQ/gain/speed so the shortcut
           "same recording channel" is not enough.
  mask   : data2vec-style - EMA teacher sees the clean view, student sees time-masked spectrogram and regresses the
           teacher's (instance-normalised) frame features at masked frames.
  distill: optional, cosine regression of a projection of the student clip embedding onto precomputed Perch v2
           embeddings (if --perch npz given; aligned to the song index rows).
Monitoring: logistic-regression probe of the frozen clip embedding on the labelled clips (fold 0 held out).
"""
import os, sys, json, time, math, argparse, copy, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from model import build_model, count_params
from augment.random_eq import random_eq as rand_eq
from augment.speed_perturb import speed_perturb
from losses.supcon import supcon
from training.checkpoint import load_encoder_state
from training.defaults import ws
from training.data import load_seg
from config import CLASSES, C2I, WS, SR
from training.metrics import metrics

N = 128000


class SongViews(torch.utils.data.Dataset):
    def __init__(self, songs, noise, args):
        self.s = songs.reset_index(drop=True); self.noise = noise; self.a = args
        self.by_rec = self.s.groupby('rec').indices
        self.recs = list(self.by_rec.keys()); self.rec_id = {r: i for i, r in enumerate(self.recs)}

    def __len__(self): return 10 ** 9

    def crop(self, i):
        r = self.s.iloc[i]
        t0 = r.seg_t0 - np.random.uniform(0.2, 0.6)
        return load_seg(r.npy, t0, t0 + 4, N)

    def augment(self, y):
        a = self.a
        if np.random.rand() < 0.5: y = speed_perturb(y, a.speed)   # polyphase resampling (was linear interpolation)
        if np.random.rand() < 0.7: y = rand_eq(y)
        if self.noise is not None and np.random.rand() < 0.8:
            nz = self.noise[np.random.randint(len(self.noise))].astype(np.float32)
            s = np.random.randint(0, len(nz) - N); nz = nz[s:s + N]
            snr = np.random.uniform(-3, 20); py, pn = (y ** 2).mean() + 1e-9, (nz ** 2).mean() + 1e-9
            y = y + nz * np.sqrt(py / pn / 10 ** (snr / 10))
        y = y * 10 ** (np.random.uniform(-10, 10) / 20)
        pk = np.abs(y).max(); return (y / pk * 0.5 if pk > 1e-6 else y).astype(np.float32)

    def __getitem__(self, _):
        rec = self.recs[np.random.randint(len(self.recs))]
        idx = self.by_rec[rec]
        i = idx[np.random.randint(len(idx))]
        j = idx[np.random.randint(len(idx))] if (len(idx) > 1 and np.random.rand() < self.a.p_other_song) else i
        y1, y2 = self.crop(i), self.crop(j)
        h = self.rec_id[rec]
        clean = y2 / max(np.abs(y2).max(), 1e-6) * 0.5
        return torch.from_numpy(self.augment(y1)), torch.from_numpy(self.augment(y2)), h, torch.from_numpy(clean.astype(np.float32))


def time_mask(S, ratio=0.4, span=12):
    B, C, Fq, T = S.shape
    m = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        while m[b].float().mean() < ratio:
            t0 = np.random.randint(0, T - span); m[b, t0:t0 + span] = True
    S = S.masked_fill(m[:, None, None, :], 0.0)
    return S, m


def probe(model, data, folds=(0,)):
    """frozen-embedding logistic probe on labelled clips (macro-F1 on held-out fold 0)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    model.eval(); E = []
    with torch.no_grad():
        for i in range(0, len(data), 64):
            X = np.stack([load_seg(r.npy, r.t0, r.t1, N) for r in data.iloc[i:i + 64].itertuples()])
            X = X / np.maximum(np.abs(X).max(1, keepdims=True), 1e-6) * 0.5
            h = model.frames(torch.from_numpy(X.astype(np.float32))); p, _ = model.pool(h); E.append(model.emb(p).numpy())
    E = np.concatenate(E); y = data.label.map(C2I).values; out = {}
    for k in folds:
        tr = (data.fold != k).values; va = ((data.fold == k) & ~data.aug).values
        sc = StandardScaler().fit(E[tr]); clf = LogisticRegression(C=0.5, max_iter=2000, class_weight='balanced').fit(sc.transform(E[tr]), y[tr])
        m = metrics(y[va], clf.predict_proba(sc.transform(E[va]))); out[k] = (m['macro_f1'], m['dialect_f1_on_dialect_clips'])
    model.train(); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--threads', type=int, default=24); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--steps', type=int, default=20000); ap.add_argument('--bs', type=int, default=64); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--w_supcon', type=float, default=1.0); ap.add_argument('--w_mask', type=float, default=1.0); ap.add_argument('--w_distill', type=float, default=0.0)
    ap.add_argument('--w_hubert', type=float, default=0.0); ap.add_argument('--codebook', default='')
    ap.add_argument('--w_mae', type=float, default=0.0); ap.add_argument('--w_ae', type=float, default=0.0)
    ap.add_argument('--w_srcpred', type=float, default=0.0); ap.add_argument('--teacher_clean', type=int, default=0)
    ap.add_argument('--tau', type=float, default=0.1); ap.add_argument('--ema', type=float, default=0.999); ap.add_argument('--speed', type=float, default=0.02)
    ap.add_argument('--p_other_song', type=float, default=0.8); ap.add_argument('--min_yh', type=float, default=0.3)
    ap.add_argument('--widths', type=int, nargs='+', default=[16, 32, 64, 128]); ap.add_argument('--d', type=int, default=160)
    ap.add_argument('--sources', default='ncs,vizina,audiomoth_bhbl,birdnet_yh'); ap.add_argument('--probe_every', type=int, default=2000)
    ap.add_argument('--exclude_test', action='store_true', default=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out_dir = f'{WS}/ssl/{args.name}'; os.makedirs(out_dir, exist_ok=True); json.dump(vars(args), open(f'{out_dir}/args.json', 'w'), indent=1)
    S = pd.read_parquet(f'{WS}/data/song_index.parquet')
    S = S[S.source.isin(args.sources.split(',')) & ((S.yh >= args.min_yh) | S.yh.isna())]
    S = S[(S.seg_t1 - S.seg_t0) <= 3.8]
    if args.exclude_test: S = S[S.split == 'train']  # held-out / uncoded Vizina males never seen, even label-free
    # Vizina files naming exactly one male: positives = that male's songs across recordings/years (kills channel shortcut)
    vf = pd.read_csv(f'{WS}/data/vizina_files.csv', dtype={'inds': str})
    one = vf[vf.inds.fillna('').str.count(';') == 0].dropna(subset=['inds'])
    ind_by = dict(zip(one.npy, one.inds))
    m = S.npy.map(ind_by)
    S = S.copy(); S.loc[m.notna(), 'rec'] = 'vind:' + m[m.notna()]
    print('vizina songs regrouped by individual:', int(m.notna().sum()), 'individuals', m.nunique(), flush=True)
    print('songs', len(S), 'recs', S.rec.nunique(), S.source.value_counts().to_dict(), flush=True)
    noise = list(np.load(f'{WS}/data/noise_bank_audiomoth.npy', allow_pickle=True))
    data = pd.read_csv(f'{WS}/data/orig_index.csv', keep_default_na=False, na_values=[''])
    ds = SongViews(S, noise, args)
    dl = iter(torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=args.workers, persistent_workers=True, prefetch_factor=4))
    model = build_model('strnadnet', n_classes=len(CLASSES), widths=tuple(args.widths), d=args.d)
    model = model.to(memory_format=torch.channels_last)
    print('params', count_params(model), flush=True)
    proj = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128))
    pred = nn.Conv1d(args.d, args.d, 1)
    src_head = nn.Sequential(nn.Linear(128, 128, bias=False), nn.Linear(128, len(ds.recs), bias=False))
    # MAE decoder: student frame features (B,d,T') -> pooled log-mel (B,F,T') at masked frames
    mae_dec = nn.Sequential(nn.Conv1d(args.d, 256, 3, padding=1), nn.GELU(), nn.Conv1d(256, 128, 3, padding=1))
    # bottleneck AE decoder: clip embedding (128) -> whole pooled log-mel (128 x T')
    class AEDec(nn.Module):
        def __init__(s, T=100):
            super().__init__(); s.T = T; s.fc = nn.Linear(128, 32 * T); s.pos = nn.Parameter(torch.zeros(1, 32, T))
            s.net = nn.Sequential(nn.Conv1d(32, 128, 5, padding=2), nn.GELU(), nn.Conv1d(128, 128, 5, padding=2), nn.GELU(), nn.Conv1d(128, 128, 3, padding=1))
        def forward(s, e): return s.net(s.fc(e).view(-1, 32, s.T) + s.pos)
    ae_dec = AEDec()
    # HuBERT-style masked unit prediction: targets = nearest codebook unit of each clean 40 ms frame
    hub_C, hub_enc, hub_head = None, None, None
    if args.w_hubert > 0:
        cbz = np.load(ws(args.codebook), allow_pickle=True); hub_C = torch.from_numpy(cbz['centroids'])
        enc_pt = str(cbz['encoder']) if 'encoder' in cbz else ''
        if enc_pt:
            hub_enc = load_encoder_state(build_model('strnadnet'), torch.load(enc_pt, map_location='cpu', weights_only=False)['state']).eval()
        hub_head = nn.Conv1d(args.d, len(hub_C), 1)
    teacher = copy.deepcopy(model); [p.requires_grad_(False) for p in teacher.parameters()]
    params = list(model.parameters()) + list(proj.parameters()) + list(pred.parameters()) + (list(src_head.parameters()) if args.w_srcpred > 0 else []) \
        + (list(mae_dec.parameters()) if args.w_mae > 0 else []) + (list(ae_dec.parameters()) if args.w_ae > 0 else []) \
        + (list(hub_head.parameters()) if args.w_hubert > 0 else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    log = []; t0 = time.time()
    for step in range(args.steps):
        x1, x2, rec, xclean = next(dl)
        S1 = model.frontend(x1).contiguous(memory_format=torch.channels_last)
        S2 = model.frontend(x2).contiguous(memory_format=torch.channels_last)
        loss = 0; parts = {}; emb_all = None
        if args.w_supcon > 0 or args.w_srcpred > 0:
            hb = model.frames(torch.cat([S1, S2])); pb, _ = model.pool(hb); emb_all = model.emb(pb)
        if args.w_supcon > 0:
            z = proj(emb_all)
            lab = torch.cat([rec, rec])
            l_sc = supcon(z, lab, args.tau); loss = loss + args.w_supcon * l_sc; parts['supcon'] = l_sc.item()
        if args.w_srcpred > 0:  # Perch-2.0-style source prediction through a low-rank bottleneck
            l_sp = F.cross_entropy(src_head(emb_all), torch.cat([rec, rec]).long()); loss = loss + args.w_srcpred * l_sp; parts['srcpred'] = l_sp.item()
        if args.w_mask > 0:
            with torch.no_grad():
                St = model.frontend(xclean).contiguous(memory_format=torch.channels_last) if args.teacher_clean else S2
                th = teacher.frames(St)  # (B,d,T)
                th = F.instance_norm(th)
            Sm, m = time_mask(S2)
            sh = pred(model.frames(Sm))
            mm = m.float()
            # frame mask is at input resolution; map to feature resolution
            Tf = sh.shape[-1]; mm = F.adaptive_max_pool1d(mm[:, None], Tf).squeeze(1)
            l_m = (F.smooth_l1_loss(sh, th, reduction='none').mean(1) * mm).sum() / mm.sum().clamp_min(1)
            loss = loss + args.w_mask * l_m; parts['mask'] = l_m.item()
        if args.w_mae > 0 or args.w_ae > 0:
            tgt = F.adaptive_avg_pool2d(S2.squeeze(1).float(), (128, 100))  # pooled normalised log-mel target (B,128,100)
            tgt = (tgt - tgt.mean((1, 2), keepdim=True)) / (tgt.std((1, 2), keepdim=True) + 1e-5)
        if args.w_mae > 0:  # masked autoencoder: reconstruct the spectrogram of masked spans from visible context
            Sm2, m2 = time_mask(S2, ratio=0.4, span=16)
            recon = F.interpolate(mae_dec(model.frames(Sm2)), size=100)
            mm2 = F.adaptive_max_pool1d(m2.float()[:, None], 100).squeeze(1)
            l_mae = (((recon - tgt) ** 2).mean(1) * mm2).sum() / mm2.sum().clamp_min(1)
            loss = loss + args.w_mae * l_mae; parts['mae'] = l_mae.item()
        if args.w_hubert > 0:
            with torch.no_grad():
                Sc = model.frontend(xclean).contiguous(memory_format=torch.channels_last)
                fr = hub_enc.frames(Sc).transpose(1, 2) if hub_enc is not None else F.avg_pool2d(Sc.float(), (1, 4)).squeeze(1).transpose(1, 2)
                fr = (fr - fr.mean(-1, keepdim=True)) / (fr.std(-1, keepdim=True) + 1e-5)
                units = torch.cdist(fr, hub_C[None].expand(len(fr), -1, -1)).argmin(-1)  # (B,100)
            Sm3, m3 = time_mask(S2, ratio=0.4, span=12)
            lg = hub_head(model.frames(Sm3))  # (B,K,100)
            mm3 = F.adaptive_max_pool1d(m3.float()[:, None], lg.shape[-1]).squeeze(1)
            ce = F.cross_entropy(lg, units[:, :lg.shape[-1]], reduction='none')
            l_hub = (ce * mm3).sum() / mm3.sum().clamp_min(1); loss = loss + args.w_hubert * l_hub; parts['hubert'] = l_hub.item()
        if args.w_ae > 0:  # bottleneck autoencoder: whole spectrogram from the 128-d clip embedding
            e2 = emb_all[len(S1):] if emb_all is not None else model.emb(model.pool(model.frames(S2))[0])
            l_ae = F.mse_loss(ae_dec(e2), tgt)
            loss = loss + args.w_ae * l_ae; parts['ae'] = l_ae.item()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 3.0); opt.step(); sched.step()
        with torch.no_grad():
            for pt, ps in zip(teacher.state_dict().values(), model.state_dict().values()):
                if pt.dtype.is_floating_point: pt.mul_(args.ema).add_(ps.detach(), alpha=1 - args.ema)
                else: pt.copy_(ps)
        if step % 50 == 0:
            print(step, {k: round(v, 4) for k, v in parts.items()}, round(time.time() - t0), flush=True)
        if (step + 1) % args.probe_every == 0 or step == args.steps - 1:
            pr = probe(teacher, data); log.append(dict(step=step + 1, probe=pr, t=time.time() - t0, **parts)); print('PROBE', log[-1], flush=True)
            torch.save(dict(state=teacher.state_dict(), args=vars(args), step=step + 1), f'{out_dir}/encoder.pt')
            json.dump(log, open(f'{out_dir}/log.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
    open(f'{WS}/ssl/{sys.argv[sys.argv.index("--name") + 1]}/done', 'w').write('ok')
