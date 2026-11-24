"""What does a trained model base its answer on?

    python -m evaluation.explain_model <run> [--per_class 6] [--deletion 400]

Works on models with the factorised head, which answer three separate questions:

    is there a Yellowhammer song?      log-odds of p(song)
    is the song complete?              log-odds of p(complete | song)
    which dialect?                     log q(dialect | complete) of the predicted dialect

For each question an **occlusion map** is computed: the spectrogram is covered patch by patch
(16 mel bins x 0.2 s, filled with each frequency's median, i.e. "background") and the drop of that
question's score is recorded. A patch is important for a question when hiding it lowers the score.
Occlusion is used rather than Grad-CAM because the model's last 2-D feature map is too coarse for
Grad-CAM to tell the terminal elements apart (research log, section 16).

Only out-of-fold clips are explained: every clip is shown to the fold model that never trained on
its recording.

Outputs:
    figs/explain/<run>/<class>.png     per class: spectrogram and the three maps for example clips,
                                       with the model's answers and its temporal attention
    figs/explain/<run>/deletion.png    the deletion test (below)
    results/explain_<run>.json         the numbers
    runs/<run>/explain/                the same figures as TensorBoard images

**Deletion test** (is the explanation faithful, and does the model read the ending?). For complete
songs that the model classifies correctly, the single most important patch for "is the song
complete?" is hidden. If the model decides completeness from the terminal elements, hiding that one
patch (2.5 % of the picture) should turn many answers into "Unfinished", while hiding a random
patch should not. The test is repeated with the three most important patches (7.5 % of the picture),
which cover a whole terminal element of any dialect.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CLASSES, DIALECTS, UNFINISHED_IDX, WS, effective_run_name
from evaluation.report import load_run_predictions
from training.checkpoint import load_model
from training.data import load_seg
from training.inference import peak_normalize

PATCH_BINS = 16  # patch height in mel bins (of 128)
PATCH_FRAMES = 20  # patch width in 10 ms frames
QUESTIONS = ['song', 'complete', 'dialect']
QUESTION_TITLES = {
    'song': 'is there a song?',
    'complete': 'is it complete?',
    'dialect': 'which dialect?',
}


# ------------------------------------------------------------------------------------ scoring
def question_scores(model, spectrograms):
    """The three scores of the factorised head for a batch of spectrograms.

    Returns:
        dict of tensors (B,): 'song' and 'complete' are log-odds, 'dialect' is the log-probability
        of the most likely dialect; plus 'probabilities' (B, 7).
    """
    outputs = model(spectrograms)

    def log_odds(log_probability):
        probability = log_probability.exp().clamp(1e-6, 1 - 1e-6)
        return torch.log(probability) - torch.log1p(-probability)

    return {
        'song': log_odds(outputs['log_song']),
        'complete': log_odds(outputs['log_complete']),
        'dialect': outputs['log_dialect'].max(1).values,
        'probabilities': torch.softmax(outputs['logits'], 1),
        'attention': outputs['att'],
    }


def patch_grid(n_bins, n_frames):
    """Top-left corners (bin, frame) of all occlusion patches."""
    return [
        (first_bin, first_frame)
        for first_bin in range(0, n_bins, PATCH_BINS)
        for first_frame in range(0, n_frames, PATCH_FRAMES)
    ]


def hide_patch(spectrogram, first_bin, first_frame):
    """A copy of a (1, F, T) spectrogram with one patch replaced by each frequency's median."""
    hidden = spectrogram.clone()
    background = spectrogram.median(dim=-1, keepdim=True).values  # (1, F, 1)
    bins = slice(first_bin, first_bin + PATCH_BINS)
    frames = slice(first_frame, first_frame + PATCH_FRAMES)
    hidden[:, bins, frames] = background[:, bins].expand(-1, -1, hidden[:, bins, frames].shape[-1])
    return hidden


@torch.no_grad()
def occlusion_maps(model, spectrogram):
    """Occlusion importance of every patch for the three questions.

    Args:
        spectrogram: (1, F, T), the model's own front-end output for one clip.

    Returns:
        ({question: (F, T) array of score drops}, scores of the intact clip)
    """
    n_bins, n_frames = spectrogram.shape[-2:]
    patches = patch_grid(n_bins, n_frames)
    batch = torch.stack([spectrogram] + [hide_patch(spectrogram, *patch) for patch in patches])
    scores = question_scores(model, batch)

    maps = {}
    for question in QUESTIONS:
        values = scores[question].cpu().numpy()
        importance = np.zeros((n_bins, n_frames), np.float32)
        for index, (first_bin, first_frame) in enumerate(patches, start=1):
            drop = values[0] - values[index]
            importance[
                first_bin : first_bin + PATCH_BINS, first_frame : first_frame + PATCH_FRAMES
            ] = drop
        maps[question] = importance
    intact = {name: value[0].cpu().numpy() for name, value in scores.items()}
    return maps, intact


# ------------------------------------------------------------------------------------ loading
class FoldModels:
    """The fold models of a run, loaded on demand."""

    def __init__(self, run, device):
        self.run_dir = f'{WS}/runs/{effective_run_name(run)}'
        self.device = device
        self.models = {}

    def get(self, fold):
        if fold not in self.models:
            model = load_model(f'{self.run_dir}/fold{fold}/model.pt').to(self.device)
            self.models[fold] = model
        return self.models[fold]


def clip_spectrogram(model, clip, device):
    """The evaluation window of a clip as the model sees it: (1, F, T) spectrogram."""
    slack = (clip.t1 - clip.t0) - 4.0
    start = clip.t0 + max(slack, 0) / 2  # the centre window, as in evaluation
    waveform = peak_normalize(load_seg(clip.npy, start, start + 4.0, 128000)[None])
    return model.frontend(torch.from_numpy(waveform).to(device))[0]


# ------------------------------------------------------------------------------------- figures
def frequency_ticks(frontend_band, n_bins, mel_scale=True):
    """Tick positions (in bins) and labels (kHz) for a spectrogram's frequency axis."""

    def to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    scale = to_mel if mel_scale else (lambda hz: hz)
    low, high = scale(frontend_band[0]), scale(frontend_band[1])
    shown = [
        khz for khz in (2, 3, 4, 5, 6, 8, 10) if frontend_band[0] <= khz * 1000 <= frontend_band[1]
    ]
    positions = [(scale(khz * 1000) - low) / (high - low) * n_bins for khz in shown]
    return positions, [str(khz) for khz in shown]


def draw_explanation_row(axes, spectrogram, maps, intact, clip, frontend_band):
    """One clip: spectrogram, then the three occlusion maps laid over it."""
    picture = spectrogram[0].cpu().numpy()
    n_bins = picture.shape[0]
    extent = [0, 4.0, 0, n_bins]  # seconds x filterbank bins; ticks are labelled in kHz below
    probabilities = intact['probabilities']
    predicted = CLASSES[int(probabilities.argmax())]
    mark = 'correct' if predicted == clip.label else 'WRONG'

    axes[0].imshow(
        picture, origin='lower', aspect='auto', extent=extent, cmap='magma', vmin=0, vmax=2
    )
    attention = intact['attention']
    time_axis = np.linspace(0, 4.0, len(attention))
    low, high = extent[2], extent[3]
    axes[0].plot(
        time_axis,
        low + (high - low) * 0.25 * attention / max(attention.max(), 1e-9),
        color='c',
        lw=1,
    )
    axes[0].set_title(
        f'label {clip.label} | model {predicted} ({probabilities.max():.2f}) {mark}',
        fontsize=8,
        loc='left',
    )

    largest = max(float(np.abs(maps[question]).max()) for question in QUESTIONS)
    for ax, question in zip(axes[1:], QUESTIONS):
        ax.imshow(
            picture, origin='lower', aspect='auto', extent=extent, cmap='gray', vmin=0, vmax=2
        )
        ax.imshow(
            maps[question],
            origin='lower',
            aspect='auto',
            extent=extent,
            cmap='bwr',
            vmin=-largest,
            vmax=largest,
            alpha=0.55,
        )
        if question == 'song':
            answer = f'p = {1 / (1 + np.exp(-intact["song"])):.2f}'
        elif question == 'complete':
            answer = f'p = {1 / (1 + np.exp(-intact["complete"])):.2f}'
        else:
            dialect_probabilities = probabilities[[CLASSES.index(name) for name in DIALECTS]]
            answer = DIALECTS[int(dialect_probabilities.argmax())]
        ax.set_title(f'{QUESTION_TITLES[question]}  {answer}', fontsize=8, loc='left')
    tick_positions, tick_labels = frequency_ticks(frontend_band, n_bins)
    for ax in axes:
        ax.set_yticks(tick_positions, tick_labels)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel('kHz', fontsize=6)


def explanation_sheets(run, predictions, fold_models, per_class, out_dir, board):
    """One sheet per class: `per_class` correct and up to `per_class` wrong clips with that label."""
    results = {}
    for class_name in CLASSES:
        of_class = predictions[predictions.label == class_name]
        correct = of_class[of_class.correct].sample(
            min(per_class, int(of_class.correct.sum())), random_state=0
        )
        wrong = of_class[~of_class.correct]
        wrong = wrong.sample(min(per_class, len(wrong)), random_state=0)
        chosen = list(correct.itertuples()) + list(wrong.itertuples())
        if not chosen:
            continue

        figure, axes = plt.subplots(len(chosen), 4, figsize=(20, 2.1 * len(chosen)), squeeze=False)
        for row, clip in zip(axes, chosen):
            model = fold_models.get(clip.cv_fold)
            spectrogram = clip_spectrogram(model, clip, fold_models.device)
            maps, intact = occlusion_maps(model, spectrogram)
            band = (model.frontend.bank.fmin, model.frontend.bank.fmax)
            draw_explanation_row(row, spectrogram, maps, intact, clip, band)
        figure.suptitle(
            f'{run}: clips labelled {class_name}. Red = hiding this patch lowers the answer, '
            'blue = raises it. Cyan line = temporal attention.',
            fontsize=10,
        )
        figure.tight_layout()
        figure.savefig(f'{out_dir}/{class_name}.png', dpi=75)
        board.add_figure(f'explain/{class_name}', figure)
        plt.close(figure)
        results[class_name] = len(chosen)
    return results


# ------------------------------------------------------------------------------- deletion test
N_HIDDEN = (1, 3)  # how many patches are hidden: one (2.5 % of the picture) and three (7.5 %)


def hide_patches(spectrogram, patches):
    hidden = spectrogram
    for first_bin, first_frame in patches:
        hidden = hide_patch(hidden, first_bin, first_frame)
    return hidden


@torch.no_grad()
def deletion_test(predictions, fold_models, n_clips, out_dir, board):
    """Hide the top "complete?" patches of correctly classified complete songs (module docstring).

    The most important one or three patches are hidden, and for comparison the same number of
    random patches. Three patches cover the 0.5 s falling C element of a BC song, which one 0.2 s
    patch does not.
    """
    complete = predictions[predictions.label.isin(DIALECTS) & predictions.correct]
    complete = complete.sample(min(n_clips, len(complete)), random_state=0)
    rng = np.random.default_rng(0)

    flipped = {(kind, n): [] for kind in ('top', 'random') for n in N_HIDDEN}
    by_dialect = {}
    relative_times = []
    for clip in complete.itertuples():
        model = fold_models.get(clip.cv_fold)
        spectrogram = clip_spectrogram(model, clip, fold_models.device)
        maps, intact = occlusion_maps(model, spectrogram)
        patches = patch_grid(*spectrogram.shape[-2:])

        importance = [
            maps['complete'][first_bin, first_frame] for first_bin, first_frame in patches
        ]
        ranked = [patches[index] for index in np.argsort(importance)[::-1]]
        shuffled = [patches[index] for index in rng.permutation(len(patches))]

        variants = [(kind, n) for kind in ('top', 'random') for n in N_HIDDEN]
        hidden = torch.stack(
            [
                hide_patches(spectrogram, (ranked if kind == 'top' else shuffled)[:n])
                for kind, n in variants
            ]
        )
        answers = question_scores(model, hidden)['probabilities'].argmax(1).cpu().numpy()
        for variant, answer in zip(variants, answers):
            flipped[variant].append(answer == UNFINISHED_IDX)
        by_dialect.setdefault(clip.label, []).append(flipped[('top', N_HIDDEN[-1])][-1])

        # where the top patch is, relative to the attention-weighted centre of the song
        attention = intact['attention']
        song_centre = float(
            (attention * np.linspace(0, 4.0, len(attention))).sum() / attention.sum()
        )
        relative_times.append((ranked[0][1] + PATCH_FRAMES / 2) * 0.01 - song_centre)

    result = {'n_clips': len(complete)}
    for (kind, n), values in flipped.items():
        result[f'unfinished_after_hiding_{n}_{kind}'] = float(np.mean(values))
    result[f'by_dialect_hiding_{N_HIDDEN[-1]}_top'] = {
        name: float(np.mean(values)) for name, values in sorted(by_dialect.items())
    }
    result['top_patch_time_minus_attention_centre_median_s'] = float(np.median(relative_times))

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6))
    positions = np.arange(len(N_HIDDEN))
    top = [result[f'unfinished_after_hiding_{n}_top'] for n in N_HIDDEN]
    random = [result[f'unfinished_after_hiding_{n}_random'] for n in N_HIDDEN]
    left.bar(positions - 0.2, top, 0.4, color='#c0392b', label='most important for "complete?"')
    left.bar(positions + 0.2, random, 0.4, color='#7f8c8d', label='random')
    left.set_xticks(positions, [f'{n} patch{"es" if n > 1 else ""} hidden' for n in N_HIDDEN])
    left.set_ylabel('share answered "Unfinished"')
    left.set_title(f'{len(complete)} correctly classified complete songs', fontsize=9)
    left.legend(fontsize=8)
    names = list(result[f'by_dialect_hiding_{N_HIDDEN[-1]}_top'])
    right.bar(
        names,
        [result[f'by_dialect_hiding_{N_HIDDEN[-1]}_top'][name] for name in names],
        color='#c0392b',
    )
    right.set_ylim(0, 1)
    right.set_title(f'by dialect, {N_HIDDEN[-1]} most important patches hidden', fontsize=9)
    figure.tight_layout()
    figure.savefig(f'{out_dir}/deletion.png', dpi=90)
    board.add_figure('explain/deletion_test', figure)
    plt.close(figure)
    return result


# ---------------------------------------------------------------------------------------- main
def main():
    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run')
    parser.add_argument('--per_class', type=int, default=6)
    parser.add_argument('--deletion', type=int, default=400, help='clips in the deletion test')
    arguments = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = effective_run_name(arguments.run)
    out_dir = f'{WS}/figs/explain/{run}'
    os.makedirs(out_dir, exist_ok=True)
    board = SummaryWriter(f'{WS}/runs/{run}/explain')

    predictions = load_run_predictions(arguments.run)
    fold_models = FoldModels(arguments.run, device)

    result = {
        'sheets': explanation_sheets(
            run, predictions, fold_models, arguments.per_class, out_dir, board
        )
    }
    result['deletion_test'] = deletion_test(
        predictions, fold_models, arguments.deletion, out_dir, board
    )
    board.close()

    with open(f'{WS}/results/explain_{run}.json', 'w') as file:
        json.dump(result, file, indent=1)
    print(json.dumps(result['deletion_test'], indent=1))
    print(f'figures in {out_dir}')


if __name__ == '__main__':
    main()
