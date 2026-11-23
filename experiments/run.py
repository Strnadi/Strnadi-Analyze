"""Run or list experiments.

    python -m experiments.run --list
    python -m experiments.run r2_factorized_weak --folds 0,1 --threads 5
    python -m experiments.run r2_factorized_weak --seed 1                       (run name gets the suffix _s1)
    python -m experiments.run frontend_pcen --set epochs=12 --set lr=1e-3       (ad-hoc overrides, parsed as Python literals)
"""
import argparse
import ast
import glob
import importlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = {'__init__', 'run', 'recipes'}


def load(stem):
    return dict(importlib.import_module(f'experiments.{stem}').EXPERIMENT)


def all_experiments():
    return sorted(os.path.basename(p)[:-3] for p in glob.glob(f'{HERE}/*.py') if os.path.basename(p)[:-3] not in SKIP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('experiment', nargs='?'); ap.add_argument('--list', action='store_true')
    ap.add_argument('--folds'); ap.add_argument('--seed', type=int); ap.add_argument('--threads', type=int); ap.add_argument('--workers', type=int)
    ap.add_argument('--name'); ap.add_argument('--set', action='append', default=[]); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    if a.list or not a.experiment:
        for stem in all_experiments():
            e = load(stem)
            print(f"{stem:34s} {e.get('origin', '?'):17s} {e.get('status', '?'):9s} {e.get('description', '').splitlines()[0][:110]}")
        return
    exp = load(a.experiment)
    if a.seed is not None: exp['seed'] = a.seed
    for kv in a.set:
        key, val = kv.split('=', 1)
        try: exp[key] = ast.literal_eval(val)
        except (ValueError, SyntaxError): exp[key] = val
    exp['name'] = a.name or exp.get('name') or (a.experiment + (f"_s{exp['seed']}" if exp.get('seed') else ''))
    runner = exp.get('runner', 'cv')
    if runner != 'cv':                                   # stand-alone training scripts with their own arguments
        module = {'ssl': 'training.ssl_pretrain', 'unsupervised': 'training.train_unsupervised', 'mil': 'training.train_mil',
                  'identity': 'training.train_identity'}[runner]
        cmd = [sys.executable, '-W', 'ignore', '-m', module] + exp['args'].format(name=exp['name'], threads=a.threads or 4).split()
        print(' '.join(cmd))
        if not a.dry: subprocess.run(cmd, check=True)
        return
    from training.defaults import make_config
    from training.trainer import train
    cfg = make_config(exp)
    if a.folds: cfg['folds'] = [int(k) for k in a.folds.split(',')]
    if a.threads: cfg['threads'] = a.threads
    if a.workers is not None: cfg['workers'] = a.workers
    if a.dry:
        print({k: v for k, v in cfg.items() if k not in ('description', 'hypothesis', 'result')}); return
    train(cfg)


if __name__ == '__main__':
    main()
