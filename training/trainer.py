"""Recording-grouped cross-validation trainer (refactor of the research session's `train_cv.py`).

Protocol per fold k:
- outer test fold = strong clips with fold == k; never used for any choice;
- inner validation = a grouped subset (`inner_val`) of the TRAINING recordings, scored every epoch;
  weak / unlabelled windows of inner-validation recordings are excluded from training as well;
- the EMA weights are what is evaluated; the checkpoint rule (`select`) picks among epochs on the inner set:
    inner_smooth3  best 3-epoch moving average of inner macro-F1 (default: argmax occasionally picks a fluke)
    inner_argmax   best single epoch (research runs base_v3 .. R6)
    last           fixed schedule, no selection (both reviews: equal or better, and with inner_val=0 it frees
                   15 % of a training set that has 79 BlBh clips)
- a resume file is written every epoch, so a crash or reboot continues instead of restarting.
Outputs under $STRNADI_WS/runs/<name>/fold<k>/: model.pt, last.pt, oof.npz, log.json.
"""
import copy
import json
import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import CLASSES, C2I, WS
from model import build_model, count_params
from layers.spec_augment import SpecAugment
from layers.freq_mixstyle import FreqMixStyle
from layers.distill_projection import DistillProjection
from augment.pipeline import WaveAugment
from losses.soft_cross_entropy import soft_cross_entropy
from losses.partial_label import partial_label_nll
from losses.factorized_weak import factorized_weak_loss
from losses.bag_complete import bag_complete_nll
from losses.margin import cosface_margin
from losses.fixmatch import FreeMatch
from losses.distill_cosine import distill_cosine
from losses.frame_bce import frame_bce
from losses.frame_ce import frame_ce
from losses.measurement import measurement_loss
from .defaults import ws
from .data import read_table, ClipDataset, UnlabeledDataset, BagDataset, EmbeddingStore
from .sampler import pool_sampling_weights
from .metrics import metrics
from .ema import EMA
from .optim import build_optimizer
from .checkpoint import save_model


# ----------------------------------------------------------------------------------------------- data
def split_tables(cfg, k):
    data = read_table(cfg['strong'])
    aug = data['aug'] if 'aug' in data else pd.Series(False, index=data.index)
    tr = data[(data.fold != k) & ~aug]                                  # pre-baked `aug_*` copies are never used
    va = data[(data.fold == k) & ~aug]
    groups = tr.grp.unique()
    inner = set(np.random.default_rng(1000 + k).choice(groups, size=int(len(groups) * cfg['inner_val']), replace=False))
    iv = tr[tr.grp.isin(inner)]
    tr = tr[~tr.grp.isin(inner)].assign(pool='strong')
    if 'train_label' in tr:        # relabelled training targets (e.g. disputed clips as candidate sets); evaluation keeps `label`
        tr['label'] = tr.train_label.fillna(tr.label)

    def usable(t):                                                      # drop rows of the test fold and of inner-validation recordings
        t = t[t.fold != k] if 'fold' in t else t
        return t[~t.rec.astype(str).str.replace('ncs:', '', regex=False).isin(inner)] if 'rec' in t else t

    for path in cfg['extra']:
        ex = usable(read_table(path.replace('{k}', str(k))))
        tr = pd.concat([tr, ex.assign(pool=os.path.basename(path).replace('{k}', '').replace('.csv', '').rstrip('_f'))], ignore_index=True)
    if 'w' in tr:
        tr['w'] = tr['w'].fillna(1.0)
    unl = usable(read_table(cfg['unlabeled'])) if cfg['unlabeled'] else None
    bags = usable(read_table(cfg['bags']['table'])) if cfg['bags'] else None
    return tr.reset_index(drop=True), iv, va, unl, bags


def load_bank(path):
    return list(np.load(ws(path), allow_pickle=True)) if path else None


# ----------------------------------------------------------------------------------------------- losses
def classification_loss(out, tgt, kind, w, cfg):
    logits = out['logits']
    if cfg['margin'] > 0:
        logits = cosface_margin(logits, tgt * (kind[:, None] != 1).float(), cfg['margin'], cfg['margin_scale'])
    hard, part = (kind == 0) | (kind == 2), kind == 1
    factor_weak = bool(part.any()) and 'log_dialect' in out
    per = torch.zeros(len(logits))
    if hard.any():
        per[hard] = soft_cross_entropy(logits[hard], tgt[hard], cfg['label_smoothing'], cfg['focal'])
    if part.any() and not factor_weak:
        per[part] = partial_label_nll(logits[part], tgt[part])
    valid = (hard if factor_weak else (hard | part)).float()
    loss = (per * w * valid).sum() / (w * valid).sum().clamp_min(1e-6)
    if factor_weak:
        sub = {key: out[key][part] for key in ('log_song', 'log_complete', 'log_dialect')}
        loss = loss + cfg['w_weak'] * factorized_weak_loss(sub, tgt[part], w[part])
    return loss


@torch.no_grad()
def evaluate(model, df, cfg, input_store=None, bs=64):
    model.eval()
    dl = torch.utils.data.DataLoader(ClipDataset(df, cfg['win'], False, cfg, input_store=input_store), batch_size=bs, num_workers=min(2, cfg['workers']))
    return np.concatenate([torch.softmax(model(b['wave'])['logits'], 1).numpy() for b in dl])


# ----------------------------------------------------------------------------------------------- one fold
def train_fold(cfg, k):
    seed = cfg['seed'] + k
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    tr, iv, va, unl, bag_rows = split_tables(cfg, k)
    out_dir = f"{WS}/runs/{cfg['name']}/fold{k}"; os.makedirs(out_dir, exist_ok=True)

    noise, species = load_bank(cfg['noise']), load_bank(cfg['species'])
    augment = WaveAugment(cfg, noise, species, far_csv=f'{WS}/data/farfield_stats.csv')
    input_store = EmbeddingStore(spatial=cfg['inputs'] == 'perch_spatial') if cfg['inputs'] != 'wave' else None
    teacher_store = EmbeddingStore() if (cfg['w_distill'] > 0 and cfg['teacher'] == 'cache') else None

    sw, shares = pool_sampling_weights(tr, cfg['pool_share'], cfg['balance'])
    print('pool shares', shares, flush=True)
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sw, dtype=torch.double), num_samples=cfg['epoch_size'], replacement=True)
    loader_kw = dict(num_workers=cfg['workers'], drop_last=True, persistent_workers=cfg['workers'] > 0)
    dl = torch.utils.data.DataLoader(ClipDataset(tr, cfg['win'], True, cfg, augment, teacher_store, input_store), batch_size=cfg['bs'], sampler=sampler, **loader_kw)

    def endless(ds, bs):                                               # replacement sampler: never exhausts (R4 once crashed on StopIteration)
        samp = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=10 ** 9)
        return iter(torch.utils.data.DataLoader(ds, batch_size=bs, sampler=samp, **loader_kw))

    udl = endless(UnlabeledDataset(unl, cfg['win'], augment), int(cfg['bs'] * cfg['u_ratio'])) if unl is not None else None
    bdl = endless(BagDataset(bag_rows, cfg['win'], augment, cfg['bags'].get('k', 6), tuple(cfg['bags'].get('sources', ('ncs',)))),
                  cfg['bags'].get('n_bags', 4)) if (bag_rows is not None and cfg['w_bag'] > 0) else None
    free = FreeMatch(len(CLASSES), cfg['tau_min']) if udl is not None else None

    model = build_model(cfg['model'], n_classes=len(CLASSES), **cfg['model_kw']).to(memory_format=torch.channels_last)
    print('model', cfg['model'], 'params', count_params(model), flush=True)
    if cfg['init']:
        sd = torch.load(ws(cfg['init']), map_location='cpu', weights_only=False)['state']
        print('init', cfg['init'], model.load_state_dict({n: v for n, v in sd.items() if not n.startswith('cls.')}, strict=False), flush=True)
    wave_input = getattr(model, 'wave_input', False) or model.frontend is None
    spec_aug = SpecAugment(**cfg['spec_aug'], fill=getattr(model, 'spec_fill', 'zero')).train()
    mixstyle = FreqMixStyle(cfg['mixstyle_p']).train() if cfg['mixstyle_p'] > 0 else None

    teacher = None
    if cfg['teacher_run'] and (cfg['w_kd'] > 0 or (cfg['w_distill'] > 0 and cfg['teacher'] == 'online')):
        from .inference import TorchEnsemble
        teacher = TorchEnsemble(ws(os.path.join('runs', cfg['teacher_run'])) if not os.path.isabs(cfg['teacher_run']) else cfg['teacher_run'])
    distill = None
    if cfg['w_distill'] > 0:
        with torch.no_grad():
            probe = model.eval()(torch.zeros(2, int(cfg['win'] * 32000)) if cfg['inputs'] == 'wave' else next(iter(dl))['wave'][:2])
            tdim = teacher.models[0](torch.zeros(2, int(cfg['win'] * 32000)))['emb'].shape[1] if (teacher and cfg['teacher'] == 'online') else 1536
        distill = DistillProjection(probe['emb'].shape[1], tdim)

    ema = EMA(model, cfg['ema'])
    fwd = torch.compile(model) if cfg['compile'] else model
    params = list(distill.parameters()) if distill is not None else []
    steps_per_epoch = cfg['epoch_size'] // cfg['bs']
    opt, sched = build_optimizer(model, params, cfg, cfg['epochs'] * steps_per_epoch)
    all_params = [p for g in opt.param_groups for p in g['params']]

    def forward(x, train_aug=True):
        if wave_input:
            return fwd(x)
        spec = model.frontend(x).contiguous(memory_format=torch.channels_last)
        if train_aug:
            spec = spec_aug(spec)
            if mixstyle is not None:
                spec = mixstyle(spec)
        return fwd(spec)

    y_va, y_iv = va.label.map(C2I).values, iv.label.map(C2I).values
    log, ep0 = [], 0
    best, best_state = (-1.0, -1), None                                # argmax rule
    best3, best3_state, prev_state = (-1.0, -1), None, None            # centred 3-epoch moving-average rule
    rpath = f'{out_dir}/resume.pt'
    if os.path.exists(rpath) and not os.path.exists(f'{out_dir}/oof.npz'):
        R = torch.load(rpath, map_location='cpu', weights_only=False)
        model.load_state_dict(R['model']); ema.model.load_state_dict(R['ema']); ema.step = R['step']
        opt.load_state_dict(R['opt']); sched.load_state_dict(R['sched'])
        if distill is not None and R.get('distill') is not None:
            distill.load_state_dict(R['distill'])
        if free is not None and R.get('free') is not None:
            free.load(R['free'])
        ep0, log, best, best_state = R['ep'] + 1, R['log'], tuple(R['best']), R['best_state']
        best3, best3_state, prev_state = tuple(R['best3']), R['best3_state'], R['prev_state']
        print('resumed after epoch', R['ep'], flush=True)

    t_start = time.time()
    for ep in range(ep0, cfg['epochs']):
        model.train(); total, nb = 0.0, 0
        for batch in dl:
            x, tgt, kind, w = batch['wave'], batch['target'], batch['kind'], batch['w'].float()
            out = forward(x)
            loss = classification_loss(out, tgt, kind, w, cfg)
            if cfg['w_frames'] > 0 and 'frames' in batch and out.get('frame_logits') is not None:
                loss = loss + cfg['w_frames'] * frame_bce(out['frame_logits'], batch['frames'])
            if cfg['w_elements'] > 0 and 'element_logits' in out:
                loss = loss + cfg['w_elements'] * frame_ce(out['element_logits'], batch['elements'])
            if cfg['w_measure'] > 0 and 'measurements' in out:
                loss = loss + cfg['w_measure'] * measurement_loss(out['measurements'], batch['measure'])
            if distill is not None:
                if teacher is not None and cfg['teacher'] == 'online':   # function matching: teacher sees the same augmented view
                    with torch.no_grad():
                        t_emb = torch.stack([m(x)['emb'] for m in teacher.models]).mean(0); t_mask = None
                else:
                    t_emb, t_mask = batch['teacher_emb'], batch['teacher_mask'].float()
                loss = loss + cfg['w_distill'] * distill_cosine(distill(out['emb']), t_emb, t_mask)
            if teacher is not None and cfg['w_kd'] > 0:
                with torch.no_grad():
                    t_logp = torch.stack([torch.log_softmax(m(x)['logits'], 1) for m in teacher.models]).logsumexp(0)
                    t_p = torch.softmax(t_logp / cfg['kd_temperature'], 1)
                s_logp = torch.log_softmax(out['logits'] / cfg['kd_temperature'], 1)
                loss = loss + cfg['w_kd'] * cfg['kd_temperature'] ** 2 * -(t_p * s_logp).sum(1).mean()
            if udl is not None:
                xw, xs = next(udl)
                with torch.no_grad():
                    label, mask = free.pseudo_labels(ema.model(xw)['logits'])
                if wave_input:
                    s_out = fwd(xs)
                else:                                                    # strong view: masked twice
                    s_out = fwd(spec_aug(spec_aug(model.frontend(xs).contiguous(memory_format=torch.channels_last))))
                loss = loss + cfg['w_u'] * free.loss(s_out['logits'], label, mask)
            if bdl is not None:
                bw, bd = next(bdl)                                       # (n_bags, k, N), (n_bags,)
                b_out = forward(bw.flatten(0, 1))
                if 'log_dialect' not in b_out:
                    raise ValueError('the bag loss needs head="factorized"')
                loss = loss + cfg['w_bag'] * bag_complete_nll(b_out, bd, bw.shape[1]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(all_params, cfg['grad_clip']); opt.step(); sched.step()
            ema.update(model)
            total += loss.item(); nb += 1

        rec = dict(ep=ep, loss=total / max(nb, 1), t=time.time() - t_start)
        cur = copy.deepcopy(ema.model.state_dict())
        if len(iv):
            mi = metrics(y_iv, evaluate(ema.model, iv, cfg, input_store))
            rec.update(inner_macro_f1=mi['macro_f1'], inner_acc=mi['acc'])
            if mi['macro_f1'] > best[0]:
                best, best_state = (mi['macro_f1'], ep), cur
            scores = [r['inner_macro_f1'] for r in log if 'inner_macro_f1' in r] + [mi['macro_f1']]
            cands = [(ep - 1, float(np.mean(scores[-3:])), prev_state)] if prev_state is not None else []
            if ep == cfg['epochs'] - 1:
                cands.append((ep, float(np.mean(scores[-2:])), cur))
            for e_, s_, st_ in cands:
                if s_ > best3[0]:
                    best3, best3_state = (s_, e_), st_
            rec.update(best_ep=best[1], best3_ep=best3[1])
        if cfg['log_outer'] and len(va):                                 # diagnostics only
            mo = metrics(y_va, evaluate(ema.model, va, cfg, input_store))
            rec.update(val_macro_f1=mo['macro_f1'], val_acc=mo['acc'])
        prev_state = cur
        log.append(rec); print(k, rec, flush=True)
        json.dump(log, open(f'{out_dir}/log.json', 'w'))
        torch.save(dict(model=model.state_dict(), ema=ema.model.state_dict(), step=ema.step, opt=opt.state_dict(), sched=sched.state_dict(), ep=ep,
                        log=log, best=best, best_state=best_state, best3=best3, best3_state=best3_state, prev_state=prev_state,
                        distill=distill.state_dict() if distill is not None else None, free=free.state() if free is not None else None), rpath + '.tmp')
        os.replace(rpath + '.tmp', rpath)

    last_state = copy.deepcopy(ema.model.state_dict())
    save_model(f'{out_dir}/last.pt', last_state, cfg, cfg['epochs'] - 1)
    rule = cfg['select'] if len(iv) else 'last'
    state, epoch = {'last': (last_state, cfg['epochs'] - 1), 'inner_argmax': (best_state, best[1]),
                    'inner_smooth3': (best3_state if best3_state is not None else best_state, best3[1])}[rule]
    save_model(f'{out_dir}/model.pt', state, cfg, epoch, select=rule)
    ema.model.load_state_dict(state)
    p = evaluate(ema.model, va, cfg, input_store)
    np.savez(f'{out_dir}/oof.npz', p=p, y=y_va, idx=va.index.values, name=va.name.values, strong=cfg['strong'], eval_crop=cfg['eval_crop'],
             select=rule, epoch=epoch)
    if os.path.exists(rpath):
        os.remove(rpath)
    return metrics(y_va, p, va.src.values if 'src' in va else None)


def train(cfg):
    torch.set_num_threads(cfg['threads'])
    os.makedirs(f"{WS}/runs/{cfg['name']}", exist_ok=True)
    json.dump(cfg, open(f"{WS}/runs/{cfg['name']}/config.json", 'w'), indent=1, default=str)
    results = {}
    for k in cfg['folds']:
        if os.path.exists(f"{WS}/runs/{cfg['name']}/fold{k}/oof.npz"):
            print(f'fold {k} already finished, skipping', flush=True); continue
        results[k] = train_fold(cfg, k)
        print(f"fold {k}: macro-F1 {results[k]['macro_f1']:.3f}  dialect|complete {results[k]['dialect_acc_given_complete']}  "
              f"complete-vs-unfinished AUC {results[k]['complete_vs_unfinished_auc']}", flush=True)
    return results
