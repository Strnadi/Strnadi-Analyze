"""Every experiment file loads, has the documentation fields, uses only known configuration keys and names a
registered model that can be built with its arguments (models that would download weights are built without
them; models needing external code are only checked for their registry entry).

    python -m tests.test_experiments
"""
import torch
from experiments.run import all_experiments, load
from model import MODELS, build_model
from training.defaults import make_config

NEEDS_EXTERNAL = {'efficientat_temporal'}


def test_experiments():
    names = all_experiments(); assert len(names) > 40
    for stem in names:
        e = load(stem)
        for field in ('description', 'hypothesis', 'origin', 'status', 'result'):
            assert e.get(field), f'{stem}: missing {field}'
        assert e['origin'] in ('legacy-repo', 'research-session', 'review-fork', 'review-base'), stem
        assert e['status'] in ('done', 'running', 'queued', 'proposed'), stem
        if e.get('runner', 'cv') != 'cv':
            assert '{name}' in e['args'], stem; continue
        cfg = make_config(e)
        assert cfg['model'] in MODELS, f'{stem}: unknown model {cfg["model"]}'
        assert set(cfg['pool_share']) <= {p.split('/')[-1].replace('.csv', '') for p in cfg['extra']}, f'{stem}: pool_share names a table not in extra'
        if cfg['model'] in NEEDS_EXTERNAL:
            continue
        kw = dict(cfg['model_kw'])
        if cfg['model'] in ('timm_temporal', 'densenet121_attnpool', 'mobilenetv3_delta'):
            kw['pretrained'] = False
        if cfg['model'].startswith('perch') and not kw.get('precomputed', True):
            continue                                              # would download the Perch port
        m = build_model(cfg['model'], n_classes=7, **kw)
        if cfg['inputs'] == 'wave':
            n = int(cfg['win'] * 32000)
            assert m.eval()(torch.randn(2, n) * 0.1)['logits'].shape == (2, 7), stem
        if cfg['w_bag'] > 0 or any('|' in '' for _ in ()):
            assert kw.get('head') == 'factorized', f'{stem}: the bag loss needs the factorised head'
    print(len(names), 'experiments ok')


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_experiments()
