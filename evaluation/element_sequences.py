"""Read the element sequences a model decodes, and score them as a classifier.

    python -m evaluation.element_sequences <run> [--examples 6]

For a model trained with the CTC loss on element sequences (`losses/element_ctc.py`,
`experiments/element_sequence_ctc.py`). Every held-out clip is decoded by its fold model into a
sequence such as `phrase B C`, and the sequence is turned into a class by a fixed rule
(`class_of_sequence`). Reported:

    accuracy of the class head, of the decoded sequence (greedy, and by summed CTC likelihood
    of each class's sequences), and how often head and sequence agree
    accuracy on the clips where they agree / disagree (disagreement is a cheap "not sure" flag)
    the most frequent decoded sequences per label
    figs/element_sequences_<run>.png   example clips with the decoded tokens drawn over time
"""

import argparse
import collections
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CLASSES, WS, effective_run_name
from evaluation.explain_model import FoldModels, clip_spectrogram, frequency_ticks
from evaluation.report import load_run_predictions
from losses.element_ctc import class_log_likelihoods, class_of_sequence, decode_tokens

TOKEN_COLOURS = {
    'phrase': '#bdc3c7',
    'B': '#f1c40f',
    'C': '#e67e22',
    'E': '#2ecc71',
    'L': '#3498db',
    'H': '#9b59b6',
    'X': '#e74c3c',
}


@torch.no_grad()
def decode_held_out(predictions, fold_models):
    """Decode every held-out clip; returns the table with `tokens`, `spans`, `sequence_class`."""
    sequences, spans, likelihood_classes = [], [], []
    for clip in predictions.itertuples():
        model = fold_models.get(clip.cv_fold)
        spectrogram = clip_spectrogram(model, clip, fold_models.device)
        token_log_probs = model(spectrogram[None])['token_log_probs']
        likelihood_classes.append(CLASSES[int(class_log_likelihoods(token_log_probs).argmax())])
        clip_sequences, clip_spans = decode_tokens(token_log_probs)
        sequences.append(clip_sequences[0])
        spans.append(clip_spans[0])

    decoded = predictions.copy()
    decoded['tokens'] = [' '.join(sequence) for sequence in sequences]
    decoded['spans'] = spans
    decoded['sequence_class'] = [class_of_sequence(sequence) for sequence in sequences]
    decoded['likelihood_class'] = likelihood_classes
    return decoded


def summarize(decoded):
    """Accuracy of the head, of the sequence, and of both when they agree."""
    head_right = (decoded.predicted == decoded.label).values
    # an ending outside the table ('other') has no class among the seven: count it as incomplete
    sequence_class = decoded.sequence_class.replace('other', 'Unfinished')
    sequence_right = (sequence_class == decoded.label).values
    agree = (decoded.predicted == sequence_class).values

    result = {
        'n_clips': len(decoded),
        'accuracy_head': float(head_right.mean()),
        'share_other': float((decoded.sequence_class == 'other').mean()),
        'accuracy_sequence': float(sequence_right.mean()),
        # the class whose sequences are most likely under CTC, summed over all alignments
        'accuracy_sequence_likelihood': float((decoded.likelihood_class == decoded.label).mean()),
        'agreement': float(agree.mean()),
        'accuracy_when_agreeing': float(head_right[agree].mean()),
        'accuracy_head_when_disagreeing': float(head_right[~agree].mean()),
        'accuracy_sequence_when_disagreeing': float(sequence_right[~agree].mean()),
        'recall_sequence': {
            name: float(sequence_right[(decoded.label == name).values].mean()) for name in CLASSES
        },
        'most_frequent_sequences': {},
    }
    for name in CLASSES:
        counts = collections.Counter(decoded.tokens[decoded.label == name])
        result['most_frequent_sequences'][name] = counts.most_common(4)
    return result


def draw_examples(decoded, fold_models, per_class, path):
    """Spectrograms of example clips with the decoded tokens as coloured bars along the top."""
    rows = []
    for name in CLASSES:
        of_class = decoded[decoded.label == name]
        rows += list(of_class.sample(min(per_class, len(of_class)), random_state=1).itertuples())

    n_columns = 3
    n_rows = int(np.ceil(len(rows) / n_columns))
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(21, 2.3 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for ax, clip in zip(axes.ravel(), rows):
        ax.axis('on')
        model = fold_models.get(clip.cv_fold)
        picture = clip_spectrogram(model, clip, fold_models.device)[0].cpu().numpy()
        n_bins = picture.shape[0]
        ax.imshow(
            picture,
            origin='lower',
            aspect='auto',
            extent=[0, 4.0, 0, n_bins],
            cmap='magma',
            vmin=0,
            vmax=2,
        )
        for token, first_frame, last_frame in clip.spans:
            start, end = first_frame * 0.04, (last_frame + 1) * 0.04
            ax.axvspan(start, end, ymin=0.9, ymax=1.0, color=TOKEN_COLOURS[token])
            ax.text((start + end) / 2, n_bins * 0.95, token, ha='center', va='center', fontsize=7)
        positions, tick_labels = frequency_ticks(
            (model.frontend.bank.fmin, model.frontend.bank.fmax), n_bins
        )
        ax.set_yticks(positions, tick_labels)
        ax.tick_params(labelsize=6)
        ax.set_title(
            f'label {clip.label} | head {clip.predicted} | sequence: {clip.tokens or "(empty)"}'
            f' -> {clip.sequence_class}',
            fontsize=8,
            loc='left',
        )
    figure.tight_layout()
    figure.savefig(path, dpi=75)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('run')
    parser.add_argument('--examples', type=int, default=6, help='example clips per class')
    arguments = parser.parse_args()

    run = effective_run_name(arguments.run)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_models = FoldModels(arguments.run, device)
    decoded = decode_held_out(load_run_predictions(arguments.run), fold_models)

    result = summarize(decoded)
    print(json.dumps(result, indent=1))
    with open(f'{WS}/results/element_sequences_{run}.json', 'w') as file:
        json.dump(result, file, indent=1)
    columns = ['name', 'label', 'predicted', 'confidence', 'tokens', 'sequence_class']
    decoded[columns].to_csv(f'{WS}/results/element_sequences_{run}.csv', index=False)

    draw_examples(
        decoded, fold_models, arguments.examples, f'{WS}/figs/element_sequences_{run}.png'
    )
    print(f'saved {WS}/figs/element_sequences_{run}.png')


if __name__ == '__main__':
    main()
