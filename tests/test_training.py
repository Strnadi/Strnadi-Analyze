"""Tests of the losses, the augmentations, the sampler and every optional path of the trainer,
on a tiny synthetic workspace (random audio, random labels): no real data is needed.

    python -m tests.test_training

The trainer runs are smoke tests: they check that each loss and data path executes and writes its
outputs. They say nothing about accuracy.
"""

import copy
import json
import os
import tempfile

_WORKSPACE = tempfile.mkdtemp(prefix='strnadi_test_')
os.environ['STRNADI_WS'] = _WORKSPACE  # must be set before `config` is imported

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from config import CLASSES, SR  # noqa: E402
from evaluation.report import load_run_predictions, plot_error_examples, plot_run, summarize_run  # noqa: E402
from model.element_tagger import ElementTagger  # noqa: E402
from model.ending_zoom import EndingZoomNet  # noqa: E402
from model.perch_mlp import PerchMLP  # noqa: E402
from model.perch_temporal import PerchTemporal  # noqa: E402
from model.strnadnet import StrnadNet  # noqa: E402
from model.strnadnet_measure import MeasurementStrnadNet  # noqa: E402
from training.checkpoint import load_model  # noqa: E402
from training.fit import FoldTraining, cross_validate  # noqa: E402
from training.settings import TrainSettings  # noqa: E402

#: A very small StrnadNet, so a training step takes a fraction of a second.
TINY = dict(widths=(8, 8, 16, 16), d=32, emb_dim=32)


def tiny_settings(**overrides):
    """Two epochs of two steps on fold 0, single-threaded."""
    values = dict(
        folds=[0], epochs=2, samples_per_epoch=16, batch_size=8, threads=1, data_workers=0
    )
    values.update(overrides)
    return TrainSettings(**values)


# ------------------------------------------------------------------------- synthetic workspace
def make_workspace():
    rng = np.random.default_rng(0)
    for folder in ('audio', 'data', 'emb', 'runs', 'results'):
        os.makedirs(f'{_WORKSPACE}/{folder}', exist_ok=True)

    recordings = []
    for index in range(12):
        path = f'{_WORKSPACE}/audio/rec{index}.npy'
        np.save(path, rng.normal(0, 2000, 12 * SR).astype(np.int16))
        recordings.append(path)

    labelled, weak, unlabelled = [], [], []
    for index in range(56):
        recording = index % 12
        start = float(rng.uniform(0, 6))
        elements = [['phrase', start + 0.6, start + 2.2], ['B', start + 2.3, start + 2.7]]
        labelled.append(
            dict(
                name=f'c{index}',
                label=CLASSES[index % 7],
                npy=recordings[recording],
                t0=start,
                t1=start + 4.4,
                grp=f'F{recording}',
                fold=recording % 2,
                aug=False,
                src='ncs' if index % 5 else 'birdnet_mobile',
                song_t0=start + 0.6,
                song_t1=start + 3.0,
                elements=json.dumps(elements),
                m_f1=4.8,
                m_f2=5.2,
                m_f3=5.6,
                m_df=0.4,
                m_bw=0.4,
                m_t2=0.3,
                m_has_first=1.0,
                m_has_second=float(index % 2),
            )
        )
    for index in range(60):
        recording = index % 12
        start = float(rng.uniform(0, 6))
        dialect = ['BC', 'BE', 'BhBl', 'BlBh', 'XB'][recording % 5]
        weak.append(
            dict(
                name=f'w{index}',
                label='None' if index % 10 == 0 else f'{dialect}|Unfinished|None',
                npy=recordings[recording],
                t0=start,
                t1=start + 4.4,
                fold=recording % 2,
                aug=False,
                w=1.0,
                sw=1.0,
                source='ncs' if recording % 3 else 'vizina',
                rec=f'ncs:F{recording}',
                song_t0=start + 0.6,
                song_t1=start + 3.0,
            )
        )
        unlabelled.append(
            dict(
                name=f'u{index}',
                npy=recordings[recording],
                t0=start,
                t1=start + 4.4,
                fold=recording % 2,
                rec=f'ncs:F{recording}',
                source='ncs',
            )
        )

    pd.DataFrame(labelled).to_csv(f'{_WORKSPACE}/data/orig_index.csv', index=False)
    pd.DataFrame(weak).to_csv(f'{_WORKSPACE}/data/weak.csv', index=False)
    pd.DataFrame(unlabelled).to_csv(f'{_WORKSPACE}/data/unl.csv', index=False)

    noise_bank = np.empty(4, dtype=object)
    noise_bank[:] = [rng.normal(0, 0.05, 6 * SR).astype(np.float16) for _ in range(4)]
    np.save(f'{_WORKSPACE}/data/noise.npy', noise_bank, allow_pickle=True)

    np.savez(
        f'{_WORKSPACE}/emb/orig_perch.npz',
        emb=rng.normal(size=(56, 1536)).astype(np.float16),
        spatial=rng.normal(size=(56, 16, 1536)).astype(np.float16),
    )


def train_and_check(run, make_model, settings):
    """Train fold 0 and check that the outputs exist and are well-formed."""
    cross_validate(run, make_model, settings)
    fold_dir = f'{_WORKSPACE}/runs/{run}/fold0'
    assert os.path.exists(f'{fold_dir}/model.pt')
    assert not os.path.exists(f'{fold_dir}/resume.pt')

    held_out = np.load(f'{fold_dir}/oof.npz', allow_pickle=True)
    assert held_out['p'].shape == (len(held_out['y']), 7)
    assert np.allclose(held_out['p'].sum(1), 1, atol=1e-4)


# --------------------------------------------------------------------------------------- tests
def test_trainer_all_paths():
    make_workspace()

    # everything at once: weak pool, factorised weak loss, FixMatch, bags, frames, cached distillation
    train_and_check(
        'everything',
        lambda: StrnadNet(head='factorized', **TINY),
        tiny_settings(
            extra_tables={'data/weak.csv': 0.4},
            unlabelled_table='data/unl.csv',
            bag_table='data/weak.csv',
            bag_size=3,
            bags_per_step=2,
            bag_loss_weight=0.5,
            frame_loss_weight=1.0,
            distill_weight=0.5,
            teacher_source='cache',
            noise_bank='data/noise.npy',
            noise_probability=0.5,
            splice_probability=0.5,
            random_eq_probability=0.5,
            mixstyle_probability=1.0,
        ),
    )
    train_and_check(
        'flat_head_partial_labels',
        lambda: StrnadNet(**TINY),
        tiny_settings(extra_tables={'data/weak.csv': 0.5}, checkpoint_rule='inner_argmax'),
    )
    train_and_check(
        'no_inner_validation',
        lambda: StrnadNet(**TINY),
        tiny_settings(inner_validation_share=0.0, checkpoint_rule='last'),
    )
    train_and_check(
        'cosine_margin',
        lambda: StrnadNet(head='cosine', head_kw=dict(k=2), **TINY),
        tiny_settings(margin=0.2),
    )
    train_and_check(
        'student_of_teacher',
        lambda: StrnadNet(**TINY),
        tiny_settings(
            teacher_source='online',
            teacher_run='everything',
            distill_weight=0.5,
            logit_distill_weight=1.0,
        ),
    )
    train_and_check(
        'perch_spatial',
        lambda: PerchTemporal(precomputed=True, d=32),
        tiny_settings(inputs='perch_spatial'),
    )
    train_and_check(
        'perch_pooled',
        lambda: PerchMLP(precomputed=True, hidden=32),
        tiny_settings(inputs='perch_pooled'),
    )
    train_and_check(
        'elements',
        lambda: ElementTagger(widths=(8, 8, 16, 16), d=32, emb_dim=32),
        tiny_settings(element_loss_weight=1.0),
    )
    train_and_check(
        'measurements',
        lambda: MeasurementStrnadNet(**TINY),
        tiny_settings(measurement_loss_weight=0.5),
    )
    train_and_check(
        'waveform_input_model',
        lambda: EndingZoomNet(
            coarse_widths=(8, 8, 16, 16), zoom_widths=(8, 16, 16), d=32, emb_dim=32
        ),
        tiny_settings(),
    )


def test_checkpoint_rebuilds_the_model():
    """A checkpoint alone is enough to get the model back (class and constructor arguments stored)."""
    model = load_model(f'{_WORKSPACE}/runs/everything/fold0/model.pt')
    assert type(model).__name__ == 'StrnadNet' and model.init_args['head'] == 'factorized'
    assert model(torch.randn(2, 128000) * 0.1)['logits'].shape == (2, 7)


def test_report():
    predictions = load_run_predictions('everything')
    assert {'predicted', 'confidence', 'correct', 'p_BC'} <= set(predictions.columns)
    summary = summarize_run('everything')
    assert 0 <= summary['macro_f1'] <= 1 and 'recall_by_source' in summary
    assert plot_run('everything') is not None
    plot_error_examples('everything', n=6, columns=3)
    assert os.path.exists(f'{_WORKSPACE}/figs/run_everything.png')
    assert os.path.exists(f'{_WORKSPACE}/figs/errors_everything.png')


def test_resume():
    """A run killed after epoch 0 continues at epoch 1 instead of starting over."""
    settings = tiny_settings()

    def make_model():
        return StrnadNet(**TINY)

    interrupted = FoldTraining('resume', make_model, settings, fold=0)
    loss = interrupted._train_one_epoch()
    entry = {'ep': 0, 'loss': loss, 't': 0.0}
    weights = copy.deepcopy(interrupted.ema.model.state_dict())
    interrupted.keeper.update(0, weights, *interrupted._validate(entry))
    interrupted.log.append(entry)
    interrupted._save_resume_point(0)

    FoldTraining('resume', make_model, settings, fold=0).run()
    with open(f'{_WORKSPACE}/runs/resume/fold0/log.json') as file:
        log = json.load(file)
    assert [entry['ep'] for entry in log] == [0, 1], log


def test_losses():
    from layers.factorized_head import FactorizedHead
    from losses.bag_complete import bag_complete_nll
    from losses.frame_bce import frame_bce
    from losses.nt_xent import nt_xent
    from losses.partial_label import partial_label_nll
    from losses.soft_cross_entropy import soft_cross_entropy

    outputs = FactorizedHead(16)(torch.randn(6, 16))

    # A bag of one song: the loss is just -log p(song, complete, dialect).
    single = bag_complete_nll(outputs, torch.tensor([2] * 6), 1)
    expected = -(outputs['log_song'] + outputs['log_complete'] + outputs['log_dialect'][:, 2])
    assert torch.allclose(single, expected, atol=1e-4)

    # More songs in the bag can only make "at least one is complete" more likely.
    bags_of_three = bag_complete_nll(outputs, torch.tensor([2, 2]), 3)
    assert (bags_of_three <= single.view(2, 3).min(1).values + 1e-5).all()

    logits = torch.randn(4, 7)
    one_hot = torch.nn.functional.one_hot(torch.tensor([0, 1, 2, 3]), 7).float()
    # A candidate set with one member is ordinary cross-entropy; the full set carries no information.
    assert torch.allclose(
        partial_label_nll(logits, one_hot), soft_cross_entropy(logits, one_hot), atol=1e-5
    )
    assert (partial_label_nll(logits, torch.ones(4, 7)).abs() < 1e-5).all()

    embeddings = torch.randn(8, 16)
    near_copies = embeddings + 0.01 * torch.randn(8, 16)
    assert nt_xent(embeddings, near_copies) < nt_xent(embeddings, torch.randn(8, 16))

    all_unknown = torch.full((2, 200), -1.0)
    assert frame_bce(torch.randn(2, 100), all_unknown) == 0


def test_augmentations():
    from augment.noise_mix import mix_noise, noise_power, signal_power
    from augment.null_splice import null_splice
    from augment.pipeline import WaveAugment
    from augment.speed_perturb import speed_perturb

    rng = np.random.default_rng(0)
    np.random.seed(0)
    time = np.arange(SR) / SR

    # Quiet background, a loud "phrase", then a "terminal element" 20 dB fainter.
    song = rng.normal(0, 0.01, 128000).astype(np.float32)
    song[40000 : 40000 + SR] += 0.5 * np.sin(2 * np.pi * 5000 * time).astype(np.float32)
    song[80000 : 80000 + SR // 2] += 0.05 * np.sin(2 * np.pi * 4500 * time[: SR // 2]).astype(
        np.float32
    )
    noise = rng.normal(0, 0.1, 128000).astype(np.float32)

    assert len(speed_perturb(song)) == len(song)
    assert len(null_splice(song)) == len(song)

    # The 'ending' reference measures the faint element, not the loud phrase.
    assert signal_power(song, 'ending') < 0.05 * signal_power(song, 'band')

    # Whatever the reference, the mixed-in noise sits at the requested SNR below it.
    for reference in ('clip', 'band', 'ending'):
        added_noise = mix_noise(song, noise, 10.0, reference) - song
        achieved_snr = 10 * np.log10(
            signal_power(song, reference) / noise_power(added_noise, reference)
        )
        assert abs(achieved_snr - 10.0) < 0.5, (reference, achieved_snr)

    settings = TrainSettings(
        noise_probability=1.0, splice_probability=1.0, random_eq_probability=1.0
    )
    augmented = WaveAugment(settings, noise_bank=[noise])(song)
    assert augmented.shape == song.shape and augmented.dtype == np.float32
    assert abs(np.abs(augmented).max() - 0.5) < 1e-4


def test_sampler():
    from training.sampler import pool_sampling_weights

    table = pd.DataFrame(
        dict(
            label=['BC'] * 50 + ['Unfinished'] * 400 + ['BC|Unfinished|None'] * 900,
            pool=['strong'] * 450 + ['weak'] * 900,
        )
    )
    weights, shares = pool_sampling_weights(table, {'weak': 0.35})
    assert abs(shares['weak'] - 0.35) < 1e-6 and abs(weights.sum() - 1) < 1e-6
    # Inside its pool the rare class is drawn more often than its share of rows.
    assert weights[:50].sum() / weights[50:450].sum() > 50 / 400


if __name__ == '__main__':
    torch.set_num_threads(1)
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print('ok', name, flush=True)
