"""Figures for the written thesis (Czech labels), drawn from the runs and tables in the workspace.

    python -m tools.thesis_figures <output directory> [figure name ...]

Without names every figure is drawn. Each function draws one figure and saves `<name>.svg`.
Run `python -m tools.thesis_figures --list` for the names.
"""

import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

from config import CLASSES, NONE_IDX, SR, WS
from evaluation.explain_model import FoldModels, clip_spectrogram, frequency_ticks, occlusion_maps
from evaluation.report import load_run_predictions
from evaluation.review_sheets import draw_spectrogram, read_audio
from training.data import read_table

REFERENCE = 'strnadnet_audited_labels'
CZECH = {
    'BC': 'BC',
    'BE': 'BE',
    'BhBl': 'BhBl',
    'BlBh': 'BlBh',
    'XB': 'XB',
    'Unfinished': 'nedozpívaný',
    'None': 'bez strnada',
}
COLOURS = {
    'BC': '#d4b400',
    'BE': '#24581a',
    'BhBl': '#61c1da',
    'BlBh': '#19268e',
    'XB': '#e2231a',
    'Unfinished': '#9a9a9a',
    'None': '#000000',
}
FIGURES = {}
TITLE_PAD = 10  # points between a panel and its title


def figure(function):
    FIGURES[function.__name__] = function
    return function


def use_thesis_style(font_directory):
    """Times in the figures, as in the text, when the thesis fonts are available."""
    for name in ('times.ttf', 'timesb.ttf', 'timesi.ttf'):
        path = os.path.join(font_directory, name)
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update(
        {'font.size': 11, 'axes.titlesize': 11, 'figure.dpi': 100, 'savefig.dpi': 200}
    )
    # text as outlines: the SVG then needs no font on the machine that typesets the thesis
    plt.rcParams['svg.fonttype'] = 'path'
    # room between the text and what it describes: titles, axis labels and tick labels
    plt.rcParams.update(
        {'axes.titlepad': TITLE_PAD, 'axes.labelpad': 7, 'xtick.major.pad': 5, 'ytick.major.pad': 5}
    )


def save(fig, out, name, **savefig_arguments):
    """Write the figure as SVG: lines and text are vectors, sharp at any print size; the
    spectrogram pictures inside are embedded at 400 dpi of the drawn size (the thesis prints the
    figures smaller, so about 700 dpi on paper). With THESIS_FIGURE_PREVIEW=<directory> a small
    PNG copy is written there for a quick look."""
    fig.savefig(f'{out}/{name}.svg', dpi=400, **savefig_arguments)
    preview = os.environ.get('THESIS_FIGURE_PREVIEW')
    if preview:
        os.makedirs(preview, exist_ok=True)
        fig.savefig(f'{preview}/{name}.png', dpi=110, **savefig_arguments)
    plt.close(fig)


def labelled_window(clip):
    """Start of the 4 s that were labelled, inside the clip's table window."""
    return clip.t0 + max((clip.t1 - clip.t0) - 4.0, 0) / 2


def tidy_spectrogram_axis(ax, title, show_x=True):
    ax.set_title(title, loc='left', pad=TITLE_PAD)
    ax.set_ylabel('kHz')
    ax.set_yticks([3, 5, 7, 9])
    ax.set_xticks(np.arange(0, ax.get_xlim()[1] + 0.01, 1.0))
    ax.tick_params(labelsize=9)
    if show_x:
        ax.set_xlabel('čas (s)')


# ------------------------------------------------------------------------------------ dialects
@figure
def dialekty(out):
    """One clear example of every class."""
    predictions = load_run_predictions(REFERENCE)
    order = ['BC', 'BE', 'BhBl', 'BlBh', 'XB', 'Unfinished']
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.6))
    for ax, name in zip(axes.ravel(), order):
        of_class = predictions[
            (predictions.label == name) & predictions.correct & predictions.in_source
        ]
        clip = of_class.sort_values(f'p_{name}', ascending=False).iloc[3]
        audio = read_audio(clip.npy, labelled_window(clip), labelled_window(clip) + 4.0)
        draw_spectrogram(ax, audio, '', band_khz=(2.5, 9.5))
        tidy_spectrogram_axis(ax, CZECH[name] if name != 'Unfinished' else 'nedozpívaný zpěv')
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'dialekty')


# ---------------------------------------------------------------------------------- label audit
CLEAR_AUDIT_EXAMPLES = {'BC': 'F001699.wav.11.wav', 'BlBh': 'F002064.wav.11.wav'}


@figure
def audit_prikladu(out):
    """Clips whose label the audit corrected: a phrase with one terminal element, filed as a dialect."""
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audit = pd.read_csv(f'{repository}/label_audit/clip_label_audit.csv', keep_default_na=False)
    relabelled = audit[(audit.verdict == 'relabelled') & (audit.label_audited == 'Unfinished')]
    predictions = load_run_predictions(REFERENCE).set_index('name')

    chosen = []
    for original in ['BC', 'BE', 'BhBl', 'BlBh', 'XB', 'None']:
        names = [
            name
            for name in relabelled[relabelled.label_original == original].name
            if name in predictions.index
        ]
        candidates = predictions.loc[names]
        candidates = candidates[candidates.in_source]
        candidates = candidates.sort_values('p_Unfinished', ascending=False)
        # examples picked by eye where the second-ranked clip has another bird in the picture
        clip = (
            candidates.loc[CLEAR_AUDIT_EXAMPLES[original]]
            if original in CLEAR_AUDIT_EXAMPLES
            else candidates.iloc[1]
        )
        chosen.append((original, clip))

    fig, axes = plt.subplots(3, 2, figsize=(11, 7.6))
    for ax, (original, clip) in zip(axes.ravel(), chosen):
        start = labelled_window(clip)
        audio = read_audio(clip.npy, start - 1.0, start + 5.5)
        draw_spectrogram(ax, audio, '', band_khz=(2.5, 9.5), marks=[1.0, 5.0])
        tidy_spectrogram_axis(ax, f'původně {CZECH[original]} → nedozpívaný')
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'audit_prikladu')


# --------------------------------------------------------------------------------- architecture
@figure
def architektura(out):
    """Block diagram of the network."""
    blocks = [
        ('zvuk\n4 s, 32 kHz', '128 000 vzorků', '#f2f2f2'),
        ('log-mel\nspektrogram', '128 pásem × 400 kroků\n1,5–11 kHz, 10 ms', '#dbe9f6'),
        ('medián\npo pásmech', 'odečtení pozadí\n+ kanál frekvence', '#dbe9f6'),
        ('4 reziduální\nstupně', '16–32–64–128 kanálů\n8 pásem × 100 kroků', '#d9ead3'),
        ('frekvence\ndo kanálů', '1 024 → 160 příznaků\nna časový krok', '#d9ead3'),
        ('dilatované\nkonvoluce', '3 vrstvy v čase\ndosah ≈ 1,5 s', '#fff2cc'),
        ('pooling\ns pozorností', 'vážený průměr\na odchylka', '#fff2cc'),
        ('embedding', '128 čísel', '#fce5cd'),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis('off')
    width, gap, y = 1.28, 0.2, 2.5
    for index, (title, detail, colour) in enumerate(blocks):
        x = 0.1 + index * (width + gap)
        ax.add_patch(
            FancyBboxPatch((x, y), width, 1.3, boxstyle='round,pad=0.03', fc=colour, ec='#555555')
        )
        ax.text(
            x + width / 2, y + 0.85, title, ha='center', va='center', fontsize=10.5, weight='bold'
        )
        ax.text(x + width / 2, y + 0.3, detail, ha='center', va='center', fontsize=7.2)
        if index:
            ax.annotate(
                '',
                (x - 0.01, y + 0.65),
                (x - gap + 0.01, y + 0.65),
                arrowprops=dict(arrowstyle='->', color='#333333'),
            )

    # the factorised head under the embedding
    head_x = 0.1 + 7 * (width + gap)
    questions = [
        ('je to zpěv strnada?', 'p(zpěv)'),
        ('je zpěv úplný?', 'p(úplný | zpěv)'),
        ('které nářečí?', 'q(nářečí | úplný)'),
    ]
    for index, (question, symbol) in enumerate(questions):
        x = head_x - 6.2 + index * 2.5
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.55), 2.2, 0.95, boxstyle='round,pad=0.03', fc='#ead1dc', ec='#555555'
            )
        )
        ax.text(x + 1.1, 1.2, question, ha='center', va='center', fontsize=10, weight='bold')
        ax.text(x + 1.1, 0.8, symbol, ha='center', va='center', fontsize=9)
        ax.annotate(
            '',
            (x + 1.1, 1.53),
            (head_x + width / 2, y - 0.02),
            arrowprops=dict(arrowstyle='->', color='#333333'),
        )
    ax.text(
        head_x - 6.2,
        0.2,
        'faktorizovaná hlava: sedm tříd vzniká součinem tří odpovědí',
        fontsize=9.5,
        style='italic',
    )
    ax.text(0.1, 4.2, 'předzpracování', fontsize=9.5, color='#2f5d8a')
    ax.text(0.1 + 3 * (width + gap), 4.2, '2D konvoluční část', fontsize=9.5, color='#38761d')
    ax.text(0.1 + 5 * (width + gap), 4.2, 'časová část', fontsize=9.5, color='#a67c00')
    save(fig, out, 'architektura', bbox_inches='tight')


@figure
def sluzba(out):
    """Block diagram of the service: from an uploaded recording to one answer per song."""
    blocks = [
        ('nahrávka', 'libovolná délka\npřevod na 32 kHz', '#f2f2f2'),
        ('okna', '4 s dlouhá\nkrok 0,4 s', '#dbe9f6'),
        ('model', 'třídy, embedding\na pozornost okna', '#d9ead3'),
        ('hlasování\npozornosti', 'vrchol křivky hlasů\n= jeden zpěv', '#fff2cc'),
        ('vážený\nprůměr oken', 'třída a jistota\nkaždého zpěvu', '#fff2cc'),
        ('odpověď', 'úseky zpěvů\ns třídou a jistotou', '#fce5cd'),
    ]
    fig, ax = plt.subplots(figsize=(11, 2.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.2)
    ax.axis('off')
    width, gap, y = 1.55, 0.3, 0.4
    for index, (title, detail, colour) in enumerate(blocks):
        x = 0.1 + index * (width + gap)
        ax.add_patch(
            FancyBboxPatch((x, y), width, 1.4, boxstyle='round,pad=0.03', fc=colour, ec='#555555')
        )
        ax.text(
            x + width / 2, y + 0.95, title, ha='center', va='center', fontsize=11, weight='bold'
        )
        ax.text(x + width / 2, y + 0.35, detail, ha='center', va='center', fontsize=8.5)
        if index:
            ax.annotate(
                '',
                (x - 0.02, y + 0.7),
                (x - gap + 0.02, y + 0.7),
                arrowprops=dict(arrowstyle='->', color='#333333'),
            )
    save(fig, out, 'sluzba', bbox_inches='tight')


# -------------------------------------------------------------------------- windows into songs
@figure
def pravidla_oken(out):
    """How the rule for combining overlapping windows changes the result (same model)."""
    with open(f'{WS}/results/whole_recording.json') as file:
        rules = json.load(file)['rules']
    names = {
        'most confident window': 'nejjistější okno\n(původní služba)',
        'aligned window': 'jedno zarovnané\nokno',
        'any complete window': 'aspoň jedno okno\ns nářečím',
        'mean': 'průměr oken',
        'song-weighted mean': 'vážený průměr\noken',
    }
    fig, ax = plt.subplots(figsize=(9, 3.8))
    positions = np.arange(len(names))
    complete = [rules[rule]['recall']['BhBl'] for rule in names]
    unfinished = [rules[rule]['recall']['Unfinished'] for rule in names]
    ax.bar(
        positions - 0.2,
        complete,
        0.4,
        color=COLOURS['BhBl'],
        label='úplné zpěvy BhBl určeny správně',
    )
    ax.bar(
        positions + 0.2,
        unfinished,
        0.4,
        color=COLOURS['Unfinished'],
        label='nedozpívané zpěvy určeny správně',
    )
    for x, value in zip(positions - 0.2, complete):
        ax.text(x, value + 0.025, f'{value:.2f}'.replace('.', ','), ha='center', fontsize=9)
    ax.set_xticks(positions, names.values())
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('podíl')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'pravidla_oken')


@figure
def hlasovani(out):
    """Attention voting on half a minute of a recording of a held-out male."""
    from evaluation.whole_recording import scan_recording
    from serving.scan import centred_weight, detect_songs
    from training.checkpoint import load_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(f'{WS}/runs/strnadnet_field_ready/final/model.pt').to(device)
    songs = read_table('data/vizina_eval.csv')
    songs = songs[songs.split == 'test']

    # a 32 s stretch with at least four annotated songs, complete and unfinished
    best = None
    for npy, recording in songs.groupby('npy'):
        for start in np.arange(0, recording.t1.max(), 4.0):
            inside = recording[(recording.t0 >= start + 1) & (recording.t1 <= start + 31)]
            if len(inside) >= 4 and inside.label.nunique() == 2:
                best = (npy, start, inside)
                break
        if best:
            break
    npy, start, inside = best

    hop = 0.2
    starts, probabilities, attention_peak = scan_recording([model], npy, hop, device)
    detections = detect_songs(starts, probabilities, attention_peak, hop, NONE_IDX)
    p_song = 1 - probabilities[:, NONE_IDX]
    weights = p_song * centred_weight(attention_peak) * (p_song > 0.5)
    vote_time = starts + attention_peak
    grid = np.arange(start, start + 32, hop)
    votes = np.array([weights[np.abs(vote_time - time) <= 0.4].sum() for time in grid])

    audio = read_audio(npy, start, start + 32)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[2, 1])
    )
    draw_spectrogram(top, audio, '', band_khz=(2.5, 9.5))
    for song in inside.itertuples():
        colour = COLOURS[song.label]
        top.plot(
            [song.t0 - start, song.t1 - start],
            [9.2, 9.2],
            color=colour,
            lw=4,
            solid_capstyle='butt',
        )
        top.text(
            (song.t0 + song.t1) / 2 - start,
            8.55,
            CZECH[song.label],
            color='white',
            ha='center',
            fontsize=9,
        )
    top.set_ylabel('kHz')
    top.set_yticks([3, 5, 7, 9])
    top.set_title(
        'spektrogram; pruhy nahoře: zpěvy vyznačené ornitology', loc='left', pad=TITLE_PAD
    )
    bottom.fill_between(grid - start, votes, color='#7f7f7f', alpha=0.5, label='součet hlasů oken')
    for time, class_index, _ in detections:
        if start <= time <= start + 32:
            name = CLASSES[class_index]
            bottom.axvline(time - start, color=COLOURS[name], lw=1.5)
            bottom.text(
                time - start - 0.25,
                votes.max() * 1.06,
                CZECH[name],
                color=COLOURS[name],
                ha='right',
                fontsize=9,
            )
    bottom.set_ylim(0, votes.max() * 1.25)
    bottom.set_xlim(0, 32)
    bottom.set_xticks(np.arange(0, 33, 2))
    bottom.set_xlabel('čas (s)')
    bottom.set_ylabel('hlasy')
    bottom.set_title(
        'každé okno hlasuje pro okamžik, kde vidí konec zpěvu; vrchol = nalezený zpěv',
        loc='left',
        pad=TITLE_PAD,
    )
    bottom.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'hlasovani')


# -------------------------------------------------------------------------------- explanations
@figure
def okluze(out):
    """Occlusion maps for one clip of four dialects: where the evidence for each answer sits."""
    run = 'strnadnet_field_ready'
    predictions = load_run_predictions(run)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_models = FoldModels(run, device)
    # the third question ("which dialect?") is left out: hiding one patch changes its answer by
    # less than 0.03 nats on these clips, the model answers "incomplete" instead
    titles = {'song': 'je to zpěv strnada?', 'complete': 'je zpěv úplný?'}

    rows = ['BC', 'BE', 'BhBl', 'XB']
    fig, axes = plt.subplots(len(rows), 3, figsize=(10.5, 7.6))
    for row, name in zip(axes, rows):
        of_class = predictions[(predictions.label == name) & predictions.correct]
        clip = of_class.sort_values('confidence', ascending=False).iloc[10]
        model = fold_models.get(clip.cv_fold)
        spectrogram = clip_spectrogram(model, clip, device)
        maps, intact = occlusion_maps(model, spectrogram)
        picture = spectrogram[0].cpu().numpy()
        n_bins = picture.shape[0]
        extent = [0, 4.0, 0, n_bins]
        ticks, tick_labels = frequency_ticks(
            (model.frontend.bank.fmin, model.frontend.bank.fmax), n_bins
        )

        row[0].imshow(
            picture, origin='lower', aspect='auto', extent=extent, cmap='magma', vmin=0, vmax=2
        )
        attention = intact['attention']
        time_axis = np.linspace(0, 4.0, len(attention))
        row[0].plot(time_axis, n_bins * 0.3 * attention / attention.max(), color='c', lw=1)
        row[0].set_title(
            f'{CZECH[name]}: spektrogram a pozornost', loc='left', fontsize=10, pad=TITLE_PAD
        )
        largest = max(float(np.abs(maps[question]).max()) for question in titles)
        for ax, question in zip(row[1:], titles):
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
            ax.set_title(titles[question], loc='left', fontsize=10, pad=TITLE_PAD)
        for ax in row:
            ax.set_yticks(ticks, tick_labels)
            ax.tick_params(labelsize=8)
        row[0].set_ylabel('kHz')
    for ax in axes[-1]:
        ax.set_xlabel('čas (s)')
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'okluze')


@figure
def mapa_embeddingu(out):
    """UMAP of held-out clip embeddings, coloured by label and by label-free HDBSCAN clusters."""
    import umap
    from sklearn.cluster import HDBSCAN

    from evaluation.embedding_map import held_out_embeddings

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings, clips = held_out_embeddings('strnadnet_field_ready', 0, device)
    labels = np.asarray(clips.label, dtype=str)
    points = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=0).fit_transform(embeddings)
    clusters = HDBSCAN(min_cluster_size=12).fit_predict(points)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5))
    for name in CLASSES:
        of_class = labels == name
        left.scatter(
            points[of_class, 0],
            points[of_class, 1],
            s=7,
            color=COLOURS[name],
            label=f'{CZECH[name]} ({int(of_class.sum())})',
        )
    left.legend(fontsize=9, markerscale=2, frameon=False)
    left.set_title('obarveno podle označení', loc='left')
    right.scatter(points[:, 0], points[:, 1], s=7, c=clusters, cmap='tab10')
    right.set_title(f'shluky nalezené bez označení ({len(set(clusters) - {-1})})', loc='left')
    for ax in (left, right):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'mapa_embeddingu')


# ------------------------------------------------------------------------------------ results
#: The run whose numbers the thesis reports.
FINAL = 'strnadnet_field_ready'
CLASS_ORDER = ['BC', 'BE', 'BhBl', 'BlBh', 'XB', 'Unfinished', 'None']


@figure
def matice_zamen(out):
    """Out-of-fold confusion matrix of the final recipe: counts, shaded by the share of the row."""
    predictions = load_run_predictions(FINAL)
    counts = pd.crosstab(predictions.label, predictions.predicted)
    counts = counts.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0).values
    share = counts / counts.sum(1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
    for row in range(len(CLASS_ORDER)):
        for column in range(len(CLASS_ORDER)):
            if counts[row, column]:
                colour = 'white' if share[row, column] > 0.5 else 'black'
                ax.text(
                    column,
                    row,
                    str(counts[row, column]),
                    ha='center',
                    va='center',
                    color=colour,
                    fontsize=10,
                )
    names = [CZECH[name] for name in CLASS_ORDER]
    ax.set_xticks(range(len(names)), names, rotation=35, ha='right')
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel('odpověď modelu')
    ax.set_ylabel('označení')
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'matice_zamen')


@figure
def prubeh_uceni(out):
    """Loss and macro-F1 over the epochs of the five folds; dots mark the kept checkpoints."""
    logs = []
    for fold in range(5):
        with open(f'{WS}/runs/{REFERENCE}/fold{fold}/log.json') as file:
            logs.append(pd.DataFrame(json.load(file)))
    epochs = logs[0].ep.values + 1

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    series = [
        ('loss', 'trénovací ztráta', '#9a9a9a', left),
        ('inner_loss', 'ztráta na vnitřní validaci', '#19268e', left),
        ('val_loss', 'ztráta na testovacím dílu', '#e2231a', left),
        ('inner_macro_f1', 'makro-F1 na vnitřní validaci', '#19268e', right),
        ('val_macro_f1', 'makro-F1 na testovacím dílu', '#e2231a', right),
    ]
    for column, label, colour, ax in series:
        values = np.stack([log[column].values for log in logs])
        for fold_values in values:
            ax.plot(epochs, fold_values, color=colour, lw=0.6, alpha=0.35)
        ax.plot(epochs, values.mean(0), color=colour, lw=1.8, label=label)
    for log in logs:
        kept = int(log.inner_loss.values.argmin())
        left.plot(epochs[kept], log.inner_loss.values[kept], 'o', color='#19268e', ms=4)
    left.set_yscale('log')
    left.set_yticks([0.1, 0.2, 0.5, 1, 2], ['0,1', '0,2', '0,5', '1', '2'])
    left.minorticks_off()
    left.set_ylabel('ztráta')
    right.set_ylabel('makro-F1')
    right.set_ylim(0.6, 1.0)
    for ax in (left, right):
        ax.set_xlabel('epocha')
        ax.legend(fontsize=9, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'prubeh_uceni')


#: Runs shown in the size figure: Czech name, group.
SIZE_FIGURE_RUNS = {
    'strnadnet_audited_labels': ('StrnadNet 0,9 M', 'own'),
    'size_strnadnet_190k': ('0,19 M', 'own'),
    'size_strnadnet_430k': ('0,43 M', 'own'),
    'size_strnadnet_2900k': ('2,9 M', 'own'),
    'speed_depthwise': ('separabilní konvoluce', 'own'),
    'speed_half_resolution': ('poloviční rozlišení', 'own'),
    'golf_190k_half_resolution': ('0,19 M, poloviční rozl.', 'own'),
    'golf_430k_half_resolution': ('0,43 M, poloviční rozl.', 'own'),
    'bc_resnet_tiny': ('BC-ResNet', 'small'),
    'contour_tcn': ('TCN nad konturou', 'small'),
    'fdy_crnn': ('CRNN s frekvenčně\ndynamickou konvolucí', 'small'),
    'patch_transformer': ('transformer nad výřezy', 'small'),
    'temporal_gru': ('StrnadNet + GRU', 'small'),
    'temporal_attention': ('StrnadNet + pozornost', 'small'),
    'backbone_mobilenetv3_large_imagenet': ('MobileNetV3-L', 'pretrained'),
    'backbone_efficientnet_b0_imagenet': ('EfficientNet-B0', 'pretrained'),
    'backbone_convnext_atto_imagenet': ('ConvNeXt-Atto', 'pretrained'),
    'backbone_regnety_imagenet': ('RegNetY', 'pretrained'),
    'backbone_densenet121_imagenet': ('DenseNet121', 'pretrained'),
}
GROUPS = {
    'own': ('vlastní síť a její varianty', '#e2231a', 'o'),
    'small': ('jiné malé architektury', '#19268e', 's'),
    'pretrained': ('sítě předtrénované na obrazech', '#24581a', '^'),
}


@figure
def velikost_presnost(out):
    """Macro-F1 against the number of parameters (and CPU time, when measured) of the recipes."""
    from tools.results_table import describe_run

    rows = []
    for run, (name, group) in SIZE_FIGURE_RUNS.items():
        row = describe_run(f'{WS}/runs/{run}', 'data/clips_audited.csv')
        if row and row['folds'] == 5:
            rows.append({**row, 'name': name, 'group': group})
    rows = pd.DataFrame(rows)

    scorecard_path = f'{WS}/results/model_scorecard.json'
    timed = {}
    if os.path.exists(scorecard_path):
        with open(scorecard_path) as file:
            timed = {row['run']: row['window_ms'] for row in json.load(file)}
    rows['window_ms'] = rows.run.map(timed)
    panels = [('parameters', 'počet parametrů')]
    if rows.window_ms.notna().sum() >= 5:
        panels.append(('window_ms', 'čas na jedno okno, jedno vlákno CPU (ms)'))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2), squeeze=False)
    for ax, (column, label) in zip(axes[0], panels):
        shown = rows[rows[column].notna()]
        for group, (group_label, colour, marker) in GROUPS.items():
            of_group = shown[shown.group == group]
            ax.scatter(
                of_group[column],
                of_group.macro_f1,
                color=colour,
                marker=marker,
                s=38,
                label=group_label,
                zorder=3,
            )
        for row in shown.itertuples():
            ax.annotate(
                row.name,
                (getattr(row, column), row.macro_f1),
                xytext=(8, 6),
                textcoords='offset points',
                fontsize=8,
            )
        ax.set_xscale('log')
        if column == 'parameters':
            ticks = [
                tick
                for tick in (1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7)
                if shown[column].min() * 0.8 <= tick <= shown[column].max() * 1.6
            ]
            ax.set_xticks(ticks, [f'{tick / 1e6:g} M'.replace('.', ',') for tick in ticks])
        else:
            ticks = [
                tick
                for tick in (1, 2, 5, 10, 20, 50, 100)
                if shown[column].min() * 0.8 <= tick <= shown[column].max() * 1.3
            ]
            ax.set_xticks(ticks, [str(tick) for tick in ticks])
        ax.minorticks_off()
        ax.set_xlim(shown[column].min() * 0.75, shown[column].max() * 1.9)
        ax.set_xlabel(label)
        ax.set_ylabel('makro-F1')
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0][0].legend(fontsize=9, frameon=False, loc='lower right')
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'velikost_presnost')


# -------------------------------------------------------------------------------- augmentation
def with_floor(audio):
    """The audio with faint white noise added, so that an emptied band is drawn dark, not as the
    amplified rounding error the per-frequency normalisation of the picture would make of it."""
    noise = np.random.default_rng(0).normal(0, 0.01 * float(np.std(audio)) + 1e-6, len(audio))
    return (np.asarray(audio, np.float32) + noise).astype(np.float32)


def clear_clip(name_of_class, rank=3):
    """A clean, correctly classified clip of a class and its 4 s of audio."""
    predictions = load_run_predictions(REFERENCE)
    of_class = predictions[
        (predictions.label == name_of_class) & predictions.correct & predictions.in_source
    ]
    clip = of_class.sort_values(f'p_{name_of_class}', ascending=False).iloc[rank]
    start = labelled_window(clip)
    return clip, read_audio(clip.npy, start, start + 4.0)


@figure
def augmentace(out):
    """One BC song under the augmentations used in training."""
    from augment.far_field import far_field, load_profiles
    from augment.noise_mix import draw_chunk, mix_noise
    from augment.species_mix import species_mix
    from augment.treble_cut import treble_cut

    np.random.seed(3)
    _, audio = clear_clip('BC')
    noise_bank = list(np.load(f'{WS}/data/noise_bank_clean.npy', allow_pickle=True))
    species_bank = list(np.load(f'{WS}/data/species_bank.npy', allow_pickle=True))
    song_bank = list(np.load(f'{WS}/data/background_song_bank.npy', allow_pickle=True))
    profiles = load_profiles(f'{WS}/data/farfield_stats.csv')
    farthest = max(profiles, key=lambda profile: profile['dist'])
    distant, distant_snr = far_field(audio, farthest)

    versions = [
        ('původní nahrávka (BC)', audio),
        (
            'přimíchaný hluk pozadí, 3 dB',
            mix_noise(audio, draw_chunk(noise_bank, len(audio)), 3.0, 'band'),
        ),
        (
            'simulace velké vzdálenosti',
            mix_noise(distant, draw_chunk(noise_bank, len(audio)), distant_snr, 'band'),
        ),
        ('jiný ptačí druh v pozadí', species_mix(audio, draw_chunk(species_bank, len(audio)))),
        ('druhý, tišší strnad v pozadí', species_mix(audio, draw_chunk(song_bank, len(audio)))),
        ('odříznuté výšky nad 6 kHz → nedozpívaný', with_floor(treble_cut(audio, 6000.0))),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.6))
    for ax, (title, version) in zip(axes.ravel(), versions):
        draw_spectrogram(ax, np.asarray(version, np.float32), '', band_khz=(2.5, 9.5))
        tidy_spectrogram_axis(ax, title)
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'augmentace')


@figure
def kontrafakt(out):
    """Counterfactual pairs: a complete song, and the same song with its ending removed."""
    from augment.treble_cut import treble_cut

    erased = read_table('data/erased_endings_v2.csv')
    clips = read_table('data/clips_audited.csv').set_index('name')
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.4))
    for row, original_class in zip(axes, ('XB', 'BC')):
        candidates = erased[erased.was == original_class].sort_values(
            'p_unfinished', ascending=False
        )
        pair = candidates.iloc[KONTRAFAKT_RANK[original_class]]
        original = clips.loc[pair['name'].split('erased:')[1]]
        start = labelled_window(original)
        audio = read_audio(original.npy, start, start + 4.0)
        without_ending = read_audio(pair.npy, 0.0, 4.0)
        versions = [
            (f'původní zpěv ({original_class})', audio),
            ('odříznuté výšky → nedozpívaný', with_floor(treble_cut(audio, 6000.0))),
            ('vymazané zakončení → nedozpívaný', without_ending),
        ]
        for ax, (title, version) in zip(row, versions):
            draw_spectrogram(ax, np.asarray(version, np.float32), '', band_khz=(2.5, 9.5))
            tidy_spectrogram_axis(ax, title)
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'kontrafakt')


KONTRAFAKT_RANK = {'XB': 12, 'BC': 2}

# --------------------------------------------------------------------------- element sequences
TOKEN_CZECH = {'phrase': 'fráze'}


@figure
def sekvence_elementu(out):
    """Decoded element sequences drawn over the spectrograms of four held-out clips."""
    from evaluation.element_sequences import TOKEN_COLOURS, decode_held_out

    predictions = load_run_predictions(FINAL)
    chosen = []
    for name in ['XB', 'BC', 'BhBl', 'Unfinished']:
        of_class = predictions[(predictions.label == name) & predictions.correct]
        chosen.append(
            of_class.sort_values(f'p_{name}', ascending=False).iloc[[SEQUENCE_RANK[name]]]
        )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_models = FoldModels(FINAL, device)
    decoded = decode_held_out(pd.concat(chosen), fold_models)

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6))
    for ax, clip in zip(axes.ravel(), decoded.itertuples()):
        model = fold_models.get(clip.cv_fold)
        picture = clip_spectrogram(model, clip, device)[0].cpu().numpy()
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
        ax.set_ylim(0, n_bins * 1.12)
        ax.set_xlim(0, 4.0)
        for token, first_frame, last_frame in clip.spans:
            start, end = first_frame * 0.04, (last_frame + 1) * 0.04
            ax.axvspan(start, end, ymin=1 / 1.12, ymax=1.0, color=TOKEN_COLOURS[token])
            ax.text(
                (start + end) / 2,
                n_bins * 1.06,
                TOKEN_CZECH.get(token, token),
                ha='center',
                va='center',
                fontsize=8,
            )
        ticks, tick_labels = frequency_ticks(
            (model.frontend.bank.fmin, model.frontend.bank.fmax), n_bins
        )
        ax.set_yticks(ticks, tick_labels)
        ax.tick_params(labelsize=9)
        ax.set_ylabel('kHz')
        ax.set_xlabel('čas (s)')
        sequence = ' '.join(TOKEN_CZECH.get(token, token) for token in clip.tokens.split())
        ax.set_title(f'{sequence}  →  {CZECH[clip.sequence_class]}', loc='left', pad=TITLE_PAD)
    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    save(fig, out, 'sekvence_elementu')


SEQUENCE_RANK = {'XB': 5, 'BC': 5, 'BhBl': 5, 'Unfinished': 5}


def main():
    if '--list' in sys.argv:
        print('\n'.join(FIGURES))
        return
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    use_thesis_style(os.path.join(os.path.dirname(os.path.abspath(out.rstrip('/'))), '..', 'fonts'))
    for name in sys.argv[2:] or list(FIGURES):
        FIGURES[name](out)
        print('saved', name, flush=True)


if __name__ == '__main__':
    main()
