"""Train the bird-level model (model/recording_mil.py) on bags of song embeddings with recording-level dialect
labels, and score it out of fold against plain averaging of per-song probabilities.

Input: results/song_emb_<run>.npz from evaluation/embed_songs.py (embeddings are out-of-fold already, so a
recording's songs were embedded by a model that never saw it). Folds are the recordings' CV folds.
Reports bird-level accuracy of (a) the MIL model and (b) the baseline: mean dialect probability over the songs
the song model calls complete.

Usage: python -m training.train_mil --name <name> --songs <run> [--epochs 30] [--max_songs 24]
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from config import WS, DIALECTS, DIALECT_IDX
from model import build_model


def bags(z, max_songs, rng=None):
    out = []
    for rec in np.unique(z['rec']):
        idx = np.where(z['rec'] == rec)[0]; lab = z['bag_label'][idx[0]]
        if lab in DIALECTS:
            out.append((idx, DIALECTS.index(lab), int(z['fold'][idx[0]])))
    return out


def collate(items, emb, max_songs, rng):
    B = len(items); D = emb.shape[1]
    X = np.zeros((B, max_songs, D), np.float32); M = np.zeros((B, max_songs), bool); y = np.zeros(B, np.int64)
    for b, (idx, lab, _) in enumerate(items):
        sel = idx if len(idx) <= max_songs else (rng.choice(idx, max_songs, replace=False) if rng is not None else idx[:max_songs])
        X[b, :len(sel)] = emb[sel]; M[b, :len(sel)] = True; y[b] = lab
    return torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--songs', required=True)
    ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--max_songs', type=int, default=24); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads); rng = np.random.default_rng(0); torch.manual_seed(0)
    z = np.load(f'{WS}/results/song_emb_{a.songs}.npz', allow_pickle=True); emb = z['emb']; all_bags = bags(z, a.max_songs)
    correct, base_correct, n = 0, 0, 0; out_dir = f'{WS}/runs/{a.name}'; os.makedirs(out_dir, exist_ok=True)
    for k in sorted({b[2] for b in all_bags}):
        tr = [b for b in all_bags if b[2] != k]; te = [b for b in all_bags if b[2] == k]
        model = build_model('recording_mil', d_in=emb.shape[1]); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
        counts = np.bincount([b[1] for b in tr], minlength=len(DIALECTS)); cw = torch.tensor((counts.sum() / np.maximum(counts, 1) / len(DIALECTS)) ** 0.5, dtype=torch.float32)
        for ep in range(a.epochs):
            model.train(); order = rng.permutation(len(tr))
            for i in range(0, len(order) - a.bs + 1, a.bs):
                X, M, y = collate([tr[j] for j in order[i:i + a.bs]], emb, a.max_songs, rng)
                loss = F.cross_entropy(model(X, M)['logits'][:, :len(DIALECTS)], y, weight=cw)      # the 'none' output is unused until bags without a complete song are labelled
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); torch.save(dict(state=model.state_dict(), model='recording_mil', model_kw=dict(d_in=int(emb.shape[1]))), f'{out_dir}/fold{k}.pt')
        with torch.no_grad():
            for i in range(0, len(te), 64):
                X, M, y = collate(te[i:i + 64], emb, a.max_songs, None)
                correct += int((model(X, M)['logits'][:, :len(DIALECTS)].argmax(1) == y).sum())
        for idx, lab, _ in te:                                                              # baseline: average over songs called complete
            p = z['prob'][idx]; complete = np.isin(p.argmax(1), DIALECT_IDX)
            pd_ = p[complete][:, DIALECT_IDX].mean(0) if complete.any() else p[:, DIALECT_IDX].mean(0)
            base_correct += int(pd_.argmax() == lab)
        n += len(te); print(f'fold {k}: MIL {correct / n:.3f}  mean-probability baseline {base_correct / n:.3f}  (cumulative, {n} recordings)', flush=True)
    res = dict(name=a.name, songs=a.songs, recordings=n, mil_acc=correct / n, mean_prob_acc=base_correct / n)
    json.dump(res, open(f'{out_dir}/result.json', 'w'), indent=1); print(json.dumps(res))


if __name__ == '__main__':
    main()
