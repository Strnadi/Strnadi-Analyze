"""Figures for the written thesis (Czech labels), drawn from the runs and tables in the workspace.

    python -m tools.thesis_figures <output directory> [figure name ...]

Without names every figure is drawn; `--list` prints the names. Conventions the thesis asks for:

- **one picture per panel.** A function that shows several things saves several files
  (`dialekt_BC.svg`, `dialekt_BE.svg`, ...); the thesis lays them out in a grid and writes the
  panel labels and the caption as text. A picture carries no title and no description, only what
  belongs to the plot itself: axes, ticks, legend, marks inside the data;
- **print size.** Every picture is drawn at the width it has on the page (the full text block,
  half of it or a third), so the font sizes in the code are the sizes in print;
- **vector output.** SVG with text as outlines; spectrograms inside are embedded at 400 dpi.

Diagrams (the architecture, the service) and the confusion matrix are not drawn here: they are
written in Typst in the thesis itself (`kapitoly/ml-schemata.typ`, `tools/thesis_tables.py`).
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

from config import CLASSES, NONE_IDX, SR, WS
from evaluation.explain_model import FoldModels, clip_spectrogram, frequency_ticks, occlusion_maps
from evaluation.report import load_run_predictions
from evaluation.review_sheets import draw_spectrogram, read_audio
from training.data import read_table

#: The reference recipe (training curves, clean example clips).
REFERENCE = 'strnadnet_audited_labels'
#: The run whose model the thesis explains and reports.
FINAL = os.environ.get('THESIS_FINAL_RUN', 'strnadnet_field_ready')

#: Widths of a picture on the page, in inches: the text block, half and a third of it.
FULL, HALF, THIRD = 6.3, 3.05, 2.0

CZECH = {
    'BC': 'BC',
    'BE': 'BE',
    'BhBl': 'BhBl',
    'BlBh': 'BlBh',
    'XB': 'XB',
    'Unfinished': 'nedozpívaný',
    'None': 'bez strnada',
}
#: File-name part of a class (ASCII).
SLUG = {**{name: name for name in CLASSES}, 'Unfinished': 'nedozpivany', 'None': 'bez_strnada'}
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
        {
            'font.size': 9,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'axes.labelpad': 5,
            'xtick.major.pad': 4,
            'ytick.major.pad': 4,
            'axes.linewidth': 0.7,
            'figure.dpi': 100,
        }
    )
    # text as outlines: the SVG then needs no font on the machine that typesets the thesis
    plt.rcParams['svg.fonttype'] = 'path'
    plt.rcParams['mathtext.fontset'] = 'stix'


def save(fig, out, name):
    """Write the figure as SVG: lines and text are vectors; spectrograms inside are embedded at
    400 dpi. With THESIS_FIGURE_PREVIEW=<directory> a PNG copy is written there for a quick look."""
    fig.savefig(f'{out}/{name}.svg', dpi=400)
    preview = os.environ.get('THESIS_FIGURE_PREVIEW')
    if preview:
        os.makedirs(preview, exist_ok=True)
        fig.savefig(f'{preview}/{name}.png', dpi=160)
    plt.close(fig)


def labelled_window(clip):
    """Start of the 4 s that were labelled, inside the clip's table window."""
    return clip.t0 + max((clip.t1 - clip.t0) - 4.0, 0) / 2


def spectrogram_panel(out, name, audio, marks=(), width=HALF, height=1.75):
    """One spectrogram (2.5-9.5 kHz) with its axes and nothing else."""
    audio = np.asarray(audio, np.float32)
    fig, ax = plt.subplots(figsize=(width, height))
    draw_spectrogram(ax, audio, '', band_khz=(2.5, 9.5), marks=list(marks))
    ax.set_title('')
    ax.set_ylabel('kHz')
    ax.set_yticks([3, 5, 7, 9])
    ax.set_xticks(np.arange(0, len(audio) / SR + 0.01, 1.0))
    ax.set_xlabel('čas (s)')
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.4)
    save(fig, out, name)


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


# ------------------------------------------------------------------------------------ dialects
@figure
def dialekty(out):
    """One clear example of every class: `dialekt_<class>.svg`."""
    for name in ['BC', 'BE', 'BhBl', 'BlBh', 'XB', 'Unfinished']:
        _, audio = clear_clip(name)
        spectrogram_panel(out, f'dialekt_{SLUG[name]}', audio)


# ---------------------------------------------------------------------------------- label audit
CLEAR_AUDIT_EXAMPLES = {'BC': 'F001699.wav.11.wav', 'BlBh': 'F002064.wav.11.wav'}


@figure
def audit_prikladu(out):
    """Clips the audit relabelled to "Unfinished", one per original label: `audit_<class>.svg`.

    Each shows the clip (between the dashed lines) with 1 s of its recording before and 1.5 s
    after: a phrase with one terminal element, filed as a dialect.
    """
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    audit = pd.read_csv(f'{repository}/label_audit/clip_label_audit.csv', keep_default_na=False)
    relabelled = audit[(audit.verdict == 'relabelled') & (audit.label_audited == 'Unfinished')]
    predictions = load_run_predictions(REFERENCE).set_index('name')

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
        if original in CLEAR_AUDIT_EXAMPLES:
            clip = candidates.loc[CLEAR_AUDIT_EXAMPLES[original]]
        else:
            clip = candidates.iloc[1]
        start = labelled_window(clip)
        audio = read_audio(clip.npy, start - 1.0, start + 5.5)
        spectrogram_panel(out, f'audit_{SLUG[original]}', audio, marks=[1.0, 5.0])


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
    fig, ax = plt.subplots(figsize=(FULL, 2.7))
    positions = np.arange(len(names))
    complete = [rules[rule]['recall']['BhBl'] for rule in names]
    unfinished = [rules[rule]['recall']['Unfinished'] for rule in names]
    ax.bar(positions - 0.2, complete, 0.4, color=COLOURS['BhBl'], label='úplné zpěvy BhBl')
    ax.bar(positions + 0.2, unfinished, 0.4, color=COLOURS['Unfinished'], label='nedozpívané zpěvy')
    for x, value in zip(positions - 0.2, complete):
        ax.text(x, value + 0.03, f'{value:.2f}'.replace('.', ','), ha='center', fontsize=8)
    ax.set_xticks(positions, names.values())
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0,25', '0,5', '0,75', '1'])
    ax.set_ylabel('podíl správně určených')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, out, 'pravidla_oken')


def voting_stretch(songs, n_songs=3):
    """The stretch of a held-out recording shown by the voting and detector figures: the first
    32 s with at least four annotated songs, complete and unfinished, cut to its last `n_songs`
    songs (two complete BhBl and an unfinished one), with 1 s of margin before and 1.5 s after.
    Returns (npy, start, length in seconds, the annotated songs inside)."""
    for npy, recording in songs.groupby('npy'):
        for start in np.arange(0, recording.t1.max(), 4.0):
            inside = recording[(recording.t0 >= start + 1) & (recording.t1 <= start + 31)]
            if len(inside) >= 4 and inside.label.nunique() == 2:
                shown = inside.sort_values('t0').iloc[-n_songs:]
                first, last = float(shown.t0.min()) - 1.0, float(shown.t1.max()) + 1.5
                return npy, first, last - first, shown
    raise ValueError('no stretch with four annotated songs')


@figure
def hlasovani(out):
    """Attention voting on half a minute of a recording of a held-out male.

    Top: the spectrogram with the songs the ornithologists annotated (bars). Bottom: the sum of
    the windows' votes; vertical lines are the songs the model found, with the class it gave.
    """
    from evaluation.whole_recording import scan_recording
    from serving.scan import centred_weight, detect_songs
    from training.checkpoint import load_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(f'{WS}/runs/{FINAL}/final/model.pt').to(device)
    songs = read_table('data/vizina_eval.csv')
    songs = songs[songs.split == 'test']

    npy, start, length, inside = voting_stretch(songs)

    hop = 0.2
    starts, probabilities, attention_peak = scan_recording([model], npy, hop, device)
    detections = detect_songs(starts, probabilities, attention_peak, hop, NONE_IDX)
    p_song = 1 - probabilities[:, NONE_IDX]
    weights = p_song * centred_weight(attention_peak) * (p_song > 0.5)
    vote_time = starts + attention_peak
    grid = np.arange(start, start + length, hop)
    votes = np.array([weights[np.abs(vote_time - time) <= 0.4].sum() for time in grid])

    audio = read_audio(npy, start, start + length)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(FULL, 3.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]}
    )
    draw_spectrogram(top, audio, '', band_khz=(2.5, 9.5))
    top.set_title('')
    for song in inside.itertuples():
        top.plot(
            [song.t0 - start, song.t1 - start],
            [9.15, 9.15],
            color=COLOURS[song.label],
            lw=3,
            solid_capstyle='butt',
        )
        top.text(
            (song.t0 + song.t1) / 2 - start,
            8.35,
            CZECH[song.label],
            color='white',
            ha='center',
            fontsize=8,
        )
    top.set_ylabel('kHz')
    top.set_yticks([3, 5, 7, 9])
    top.tick_params(labelsize=8)
    bottom.fill_between(grid - start, votes, color='#7f7f7f', alpha=0.5)
    for time, class_index, _ in detections:
        if start <= time <= start + length:
            name = CLASSES[class_index]
            bottom.axvline(time - start, color=COLOURS[name], lw=1.3)
            bottom.text(
                time - start - 0.3,
                votes.max() * 1.08,
                CZECH[name],
                color=COLOURS[name],
                ha='right',
                fontsize=8,
            )
    bottom.set_ylim(0, votes.max() * 1.3)
    bottom.set_xlim(0, length)
    bottom.set_xticks(np.arange(0, length + 0.01, 2))
    bottom.set_xlabel('čas (s)')
    bottom.set_ylabel('součet hlasů')
    bottom.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4, h_pad=1.0)
    save(fig, out, 'hlasovani')


# -------------------------------------------------------------------------------- explanations
#: The two questions of the factorised head that the occlusion figure shows. The third ("which
#: dialect?") is left out: hiding one patch changes its answer by less than 0.03 nats on these
#: clips, the model answers "incomplete" instead.
OCCLUSION_QUESTIONS = {'song': 'zpev', 'complete': 'uplnost'}


@figure
def okluze(out):
    """Occlusion maps of one clip of four dialects, three pictures per clip:

    `okluze_<class>_spektrogram.svg`  the spectrogram as the network sees it, with the temporal
                                      attention (cyan curve);
    `okluze_<class>_zpev.svg`         red: hiding the patch weakens "a Yellowhammer sings";
    `okluze_<class>_uplnost.svg`      red: hiding the patch weakens "the song is complete".
    """
    predictions = load_run_predictions(FINAL)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_models = FoldModels(FINAL, device)

    for name in ['BC', 'BE', 'BhBl', 'XB']:
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
        largest = max(float(np.abs(maps[question]).max()) for question in OCCLUSION_QUESTIONS)

        def panel(suffix, draw):
            fig, ax = plt.subplots(figsize=(THIRD, 1.5))
            draw(ax)
            keep = [index for index, label in enumerate(tick_labels) if label in ('2', '4', '8')]
            ax.set_yticks([ticks[index] for index in keep], [tick_labels[index] for index in keep])
            ax.set_xticks([0, 1, 2, 3, 4])
            ax.set_ylabel('kHz')
            ax.set_xlabel('čas (s)')
            ax.tick_params(labelsize=7)
            fig.tight_layout(pad=0.3)
            save(fig, out, f'okluze_{SLUG[name]}_{suffix}')

        def draw_spectrogram_and_attention(ax):
            ax.imshow(
                picture, origin='lower', aspect='auto', extent=extent, cmap='magma', vmin=0, vmax=2
            )
            attention = intact['attention']
            time_axis = np.linspace(0, 4.0, len(attention))
            ax.plot(time_axis, n_bins * 0.3 * attention / attention.max(), color='c', lw=0.9)

        def draw_map(question):
            def draw(ax):
                ax.imshow(
                    picture,
                    origin='lower',
                    aspect='auto',
                    extent=extent,
                    cmap='gray',
                    vmin=0,
                    vmax=2,
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

            return draw

        panel('spektrogram', draw_spectrogram_and_attention)
        for question, suffix in OCCLUSION_QUESTIONS.items():
            panel(suffix, draw_map(question))


@figure
def mapa_embeddingu(out):
    """UMAP of held-out clip embeddings: `mapa_oznaceni.svg` coloured by label, `mapa_shluky.svg`
    by the clusters HDBSCAN finds without labels. The class colours are those of `COLOURS`; the
    thesis prints the legend as text (`legenda-trid` in kapitoly/ml-schemata.typ)."""
    import umap
    from sklearn.cluster import HDBSCAN

    from evaluation.embedding_map import held_out_embeddings

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings, clips = held_out_embeddings(FINAL, 0, device)
    labels = np.asarray(clips.label, dtype=str)
    points = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=0).fit_transform(embeddings)
    clusters = HDBSCAN(min_cluster_size=12).fit_predict(points)

    def panel(name, draw):
        fig, ax = plt.subplots(figsize=(HALF, 2.9))
        draw(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0.3)
        save(fig, out, name)

    def by_label(ax):
        for name in CLASSES:
            of_class = labels == name
            ax.scatter(
                points[of_class, 0],
                points[of_class, 1],
                s=4,
                color=COLOURS[name],
            )

    def by_cluster(ax):
        ax.scatter(points[:, 0], points[:, 1], s=4, c=clusters, cmap='tab10')

    panel('mapa_oznaceni', by_label)
    panel('mapa_shluky', by_cluster)
    print(
        f'embedding map: {len(points)} clips, {len(set(clusters) - {-1})} clusters', file=sys.stderr
    )

    # The same clips in the embedding of a frozen bird-species model (Perch, averaged over time):
    # `mapa_perch_oznaceni.svg`, `mapa_perch_shluky.svg`.
    index = read_table('data/orig_index.csv')
    row_of = {name: row for row, name in enumerate(np.asarray(index.name, dtype=str))}
    perch = np.load(f'{WS}/emb/orig_perch.npz')['emb'].astype(np.float32)
    perch = perch[[row_of[name] for name in np.asarray(clips.name, dtype=str)]]
    points = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=0).fit_transform(perch)
    clusters = HDBSCAN(min_cluster_size=12).fit_predict(points)
    panel('mapa_perch_oznaceni', by_label)
    panel('mapa_perch_shluky', by_cluster)
    in_cluster = clusters >= 0
    purity = sum(
        pd.Series(labels[clusters == cluster]).value_counts().iloc[0]
        for cluster in set(clusters[in_cluster])
    ) / max(int(in_cluster.sum()), 1)
    print(
        f'Perch map: {len(set(clusters) - {-1})} clusters, {in_cluster.mean():.2f} of clips in '
        f'one, purity {purity:.2f}',
        file=sys.stderr,
    )


def embed_clips(run, fold, clips, device):
    """Embeddings (N, D) of the given clips by one fold model of a run (whether or not that model
    held the clips out)."""
    from config import effective_run_name
    from training.checkpoint import load_model
    from training.data import load_seg
    from training.inference import peak_normalize

    model = load_model(f'{WS}/runs/{effective_run_name(run)}/fold{fold}/model.pt').to(device)
    waveforms = np.stack(
        [
            load_seg(clip.npy, labelled_window(clip), labelled_window(clip) + 4.0, 4 * SR)
            for clip in clips.itertuples()
        ]
    )
    waveforms = torch.from_numpy(peak_normalize(waveforms))
    embeddings = []
    with torch.no_grad():
        for first in range(0, len(waveforms), 128):
            embeddings.append(model(waveforms[first : first + 128].to(device))['emb'].cpu().numpy())
    return np.concatenate(embeddings)


def map_panel(out, name, points, labels, anchors=None, limits=None):
    """One map: clips as dots in their class colours (the legend is Typst text)."""
    fig, ax = plt.subplots(figsize=(THIRD, 2.0))
    for class_name in CLASSES:
        of_class = labels == class_name
        ax.scatter(points[of_class, 0], points[of_class, 1], s=2.5, color=COLOURS[class_name])
    if anchors is not None:
        for x, y in anchors:
            ax.plot(x, y, marker='+', color='k', ms=7, mew=1.0)
    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.3)
    save(fig, out, name)


#: Three trainings of the combined recipe for the stability figure. The seed of a fold model is
#: the run's seed plus the fold, so fold 1 of the first run and fold 0 of the second start from
#: the same random weights, and fold 1 of both runs from different ones.
STABILITY_RUNS = {
    'first': ('final_combined', 1),
    'other_start': ('final_combined_s1', 1),
    'same_start': ('final_combined_s1', 0),
}


@figure
def stabilita_mapy(out):
    """Does a retrained model put the clips in the same places? The held-out clips of one fold are
    mapped with UMAP fitted on the first training (`stabilita_prvni.svg`); the same clips embedded
    by a training with other initial weights (`stabilita_jiny_start.svg`) and by one with the same
    initial weights but other training data (`stabilita_stejny_start.svg`) are placed into that
    same map. Prints the cosine between the embeddings of the same clip."""
    import umap

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run, fold = STABILITY_RUNS['first']
    predictions = load_run_predictions(run)
    clips = predictions[predictions.cv_fold == fold]
    labels = np.asarray(clips.label, dtype=str)
    embeddings = {key: embed_clips(*value, clips, device) for key, value in STABILITY_RUNS.items()}
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=0).fit(embeddings['first'])
    points = {
        'first': reducer.embedding_,
        'other_start': reducer.transform(embeddings['other_start']),
        'same_start': reducer.transform(embeddings['same_start']),
    }
    first = points['first']
    margin_x, margin_y = np.ptp(first[:, 0]) * 0.06, np.ptp(first[:, 1]) * 0.06
    limits = (
        (first[:, 0].min() - margin_x, first[:, 0].max() + margin_x),
        (first[:, 1].min() - margin_y, first[:, 1].max() + margin_y),
    )
    names = {'first': 'prvni', 'other_start': 'jiny_start', 'same_start': 'stejny_start'}
    for key, panel_points in points.items():
        map_panel(out, f'stabilita_{names[key]}', panel_points, labels, limits=limits)

    def unit(vectors):
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-6)

    for key in ('other_start', 'same_start'):
        cosine = (unit(embeddings['first']) * unit(embeddings[key])).sum(1)
        print(
            f'stability {key}: {len(labels)} clips, same-clip cosine {np.median(cosine):.2f} '
            f'(median), {cosine.mean():.2f} (mean)',
            file=sys.stderr,
        )


# ------------------------------------------------------------------------------------ training
@figure
def prubeh_uceni(out):
    """Training of the five folds: `prubeh_ztrata.svg` (losses; dots mark the kept checkpoints)
    and `prubeh_f1.svg` (macro-F1). Thin lines are folds, thick lines their mean."""
    logs = []
    for fold in range(5):
        with open(f'{WS}/runs/{REFERENCE}/fold{fold}/log.json') as file:
            logs.append(pd.DataFrame(json.load(file)))
    epochs = logs[0].ep.values + 1

    def panel(name, series, finish):
        fig, ax = plt.subplots(figsize=(HALF, 2.35))
        for column, label, colour in series:
            values = np.stack([log[column].values for log in logs])
            for fold_values in values:
                ax.plot(epochs, fold_values, color=colour, lw=0.5, alpha=0.35)
            ax.plot(epochs, values.mean(0), color=colour, lw=1.5, label=label)
        finish(ax)
        ax.set_xlabel('epocha')
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(pad=0.4)
        save(fig, out, name)

    def finish_loss(ax):
        for log in logs:
            kept = int(log.inner_loss.values.argmin())
            ax.plot(epochs[kept], log.inner_loss.values[kept], 'o', color='#19268e', ms=3)
        ax.set_yscale('log')
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2], ['0,1', '0,2', '0,5', '1', '2'])
        ax.minorticks_off()
        ax.set_ylabel('ztráta')

    def finish_f1(ax):
        ax.set_ylim(0.6, 1.0)
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0], ['0,6', '0,7', '0,8', '0,9', '1'])
        ax.set_ylabel('makro-F1')

    panel(
        'prubeh_ztrata',
        [
            ('loss', 'trénovací', '#9a9a9a'),
            ('inner_loss', 'vnitřní validace', '#19268e'),
            ('val_loss', 'testovací díl', '#e2231a'),
        ],
        finish_loss,
    )
    panel(
        'prubeh_f1',
        [
            ('inner_macro_f1', 'vnitřní validace', '#19268e'),
            ('val_macro_f1', 'testovací díl', '#e2231a'),
        ],
        finish_f1,
    )


@figure
def krivka_uceni(out):
    """Accuracy and macro-F1 against the share of labelled clips used for training, with and
    without the weakly labelled songs: `krivka_presnost.svg`, `krivka_f1.svg`."""
    from tools.results_table import describe_run

    series = {
        'jen označené výřezy': (
            '#19268e',
            'o',
            {
                10: 'learning_curve_10pct',
                25: 'learning_curve_25pct',
                50: 'learning_curve_50pct',
                100: 'ablation_clip_labels_only',
            },
        ),
        'označené výřezy a slabě označené zpěvy': (
            '#e2231a',
            's',
            {
                10: 'learning_curve_10pct_with_weak_songs',
                25: 'learning_curve_25pct_with_weak_songs',
                50: 'learning_curve_50pct_with_weak_songs',
                100: REFERENCE,
            },
        ),
    }
    results = {}
    for label, (_, _, runs) in series.items():
        results[label] = {
            share: describe_run(f'{WS}/runs/{run}', 'data/clips_audited.csv')
            for share, run in runs.items()
        }

    for name, key, axis_label, limits in [
        ('krivka_presnost', 'accuracy', 'přesnost', (0.88, 0.98)),
        ('krivka_f1', 'macro_f1', 'makro-F1', (0.64, 0.96)),
    ]:
        fig, ax = plt.subplots(figsize=(HALF, 2.5))
        for label, (colour, marker, _) in series.items():
            shares = sorted(share for share, row in results[label].items() if row)
            values = [results[label][share][key] for share in shares]
            ax.plot(shares, values, color=colour, marker=marker, ms=4, lw=1.3, label=label)
        ax.set_xscale('log')
        ax.set_xticks([10, 25, 50, 100], ['10 %', '25 %', '50 %', '100 %'])
        ax.minorticks_off()
        ax.set_ylim(*limits)
        ticks = ax.get_yticks()
        ax.set_yticks(ticks, [f'{tick:.2f}'.replace('.', ',') for tick in ticks])
        ax.set_ylim(*limits)
        ax.set_xlabel('podíl označených výřezů použitých k učení')
        ax.set_ylabel(axis_label)
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        if key == 'accuracy':
            ax.legend(frameon=False, fontsize=7.5, loc='lower right')
        fig.tight_layout(pad=0.4)
        save(fig, out, name)


# ---------------------------------------------------------------------------- size and speed
#: Runs shown in the size figure: Czech name, group.
SIZE_FIGURE_RUNS = {
    'strnadnet_audited_labels': ('výchozí 0,9 M', 'own'),
    'size_strnadnet_190k': ('0,19 M', 'own'),
    'size_strnadnet_430k': ('0,43 M', 'own'),
    'size_strnadnet_2900k': ('2,9 M', 'own'),
    'speed_depthwise': ('separabilní', 'own'),
    'speed_half_resolution': ('poloviční rozlišení', 'own'),
    'golf_190k_half_resolution': ('0,17 M', 'own'),
    'bc_resnet_tiny': ('BC-ResNet', 'small'),
    'contour_tcn': ('kontura', 'small'),
    'temporal_gru': ('GRU', 'small'),
    'temporal_attention': ('self-attention', 'small'),
    'backbone_mobilenetv3_large_imagenet': ('MobileNetV3', 'pretrained'),
    'backbone_efficientnet_b0_imagenet': ('EfficientNet-B0', 'pretrained'),
    'backbone_convnext_atto_imagenet': ('ConvNeXt-Atto', 'pretrained'),
    'backbone_regnety_imagenet': ('RegNetY', 'pretrained'),
    'backbone_densenet121_imagenet': ('DenseNet121', 'pretrained'),
}
GROUPS = {
    'own': ('vlastní síť a její varianty', '#e2231a', 'o'),
    'small': ('jiné architektury učené od nuly', '#19268e', 's'),
    'pretrained': ('sítě předtrénované na obrazech', '#24581a', '^'),
}
#: Where the label of a point goes when the default (right, above) would collide.
LABEL_OFFSETS = {
    'size_strnadnet_430k': (-8, 7),
    'speed_depthwise': (-50, -4),
    'speed_half_resolution': (6, -9),
    'temporal_gru': (-8, 7),
    'backbone_convnext_atto_imagenet': (-22, 8),
    'backbone_efficientnet_b0_imagenet': (-56, -11),
    'backbone_mobilenetv3_large_imagenet': (6, -3),
    'golf_190k_half_resolution': (-10, 7),
    'size_strnadnet_190k': (6, -9),
    'size_strnadnet_2900k': (-10, -12),
    'strnadnet_audited_labels': (4, 6),
}


@figure
def velikost_presnost(out):
    """Macro-F1 against the number of parameters (`velikost_parametry.svg`) and, once
    `tools/model_scorecard.py` has timed the models, against CPU time (`velikost_cas.svg`)."""
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
    panels = [('parameters', 'počet parametrů', 'velikost_parametry')]
    if rows.window_ms.notna().sum() >= 5:
        panels.append(('window_ms', 'čas na jedno okno (ms)', 'velikost_cas'))

    for column, label, name in panels:
        shown = rows[rows[column].notna()]
        fig, ax = plt.subplots(figsize=(FULL * 0.78, 3.2))
        for group, (group_label, colour, marker) in GROUPS.items():
            of_group = shown[shown.group == group]
            ax.scatter(
                of_group[column],
                of_group.macro_f1,
                color=colour,
                marker=marker,
                s=22,
                label=group_label,
                zorder=3,
            )
        for row in shown.itertuples():
            ax.annotate(
                row.name,
                (getattr(row, column), row.macro_f1),
                xytext=LABEL_OFFSETS.get(row.run, (6, 5)),
                textcoords='offset points',
                fontsize=7,
            )
        ax.set_xscale('log')
        candidates = (
            (1e5, 2e5, 5e5, 1e6, 2e6, 5e6, 1e7)
            if column == 'parameters'
            else (2, 5, 10, 20, 50, 100)
        )
        ticks = [
            tick
            for tick in candidates
            if shown[column].min() * 0.8 <= tick <= shown[column].max() * 1.6
        ]
        if column == 'parameters':
            ax.set_xticks(ticks, [f'{tick / 1e6:g} M'.replace('.', ',') for tick in ticks])
        else:
            ax.set_xticks(ticks, [str(tick) for tick in ticks])
        ax.minorticks_off()
        ax.set_xlim(shown[column].min() * 0.7, shown[column].max() * 2.2)
        yticks = ax.get_yticks()
        ax.set_yticks(yticks, [f'{tick:.2f}'.replace('.', ',') for tick in yticks])
        ax.set_ylim(shown.macro_f1.min() - 0.012, shown.macro_f1.max() + 0.012)
        ax.set_xlabel(label)
        ax.set_ylabel('makro-F1')
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, loc='lower right', fontsize=7.5)
        fig.tight_layout(pad=0.4)
        save(fig, out, name)


# -------------------------------------------------------------------------------- augmentation
@figure
def augmentace(out):
    """One BC song under the augmentations used in training: `augmentace_<what>.svg`."""
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
    n_samples = len(audio)

    versions = {
        'puvodni': audio,
        'hluk': mix_noise(audio, draw_chunk(noise_bank, n_samples), 3.0, 'band'),
        'vzdalenost': mix_noise(distant, draw_chunk(noise_bank, n_samples), distant_snr, 'band'),
        'jiny_druh': species_mix(audio, draw_chunk(species_bank, n_samples)),
        'druhy_strnad': species_mix(audio, draw_chunk(song_bank, n_samples)),
        'bez_vysek': with_floor(treble_cut(audio, 6000.0)),
    }
    for name, version in versions.items():
        spectrogram_panel(out, f'augmentace_{name}', version)


@figure
def test_odstraneni(out):
    """The deletion test as curves (`odstraneni_krivka.svg`): the share of correctly classified
    complete songs answered "Unfinished" after hiding the k patches most important for the
    question "complete?", against k random patches; and by dialect at k = 3
    (`odstraneni_nareci.svg`). The same 400 songs and the same random patches as
    `evaluation.explain_model`, with k up to 8; the numbers go to results/deletion_curve_<run>.json."""
    from config import DIALECTS, UNFINISHED_IDX
    from evaluation.explain_model import (
        hide_patches,
        occlusion_maps,
        patch_grid,
        question_scores,
    )

    hidden_counts = list(range(0, 9))
    predictions = load_run_predictions(FINAL)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_models = FoldModels(FINAL, device)
    complete = predictions[predictions.label.isin(DIALECTS) & predictions.correct]
    complete = complete.sample(min(400, len(complete)), random_state=0)
    rng = np.random.default_rng(0)
    flipped = {(kind, n): [] for kind in ('top', 'random') for n in hidden_counts}
    by_dialect = {}
    for clip in complete.itertuples():
        model = fold_models.get(clip.cv_fold)
        spectrogram = clip_spectrogram(model, clip, device)
        maps, _ = occlusion_maps(model, spectrogram)
        patches = patch_grid(*spectrogram.shape[-2:])
        importance = [
            maps['complete'][first_bin, first_frame] for first_bin, first_frame in patches
        ]
        ranked = [patches[index] for index in np.argsort(importance)[::-1]]
        shuffled = [patches[index] for index in rng.permutation(len(patches))]
        variants = [(kind, n) for kind in ('top', 'random') for n in hidden_counts]
        hidden = torch.stack(
            [
                hide_patches(spectrogram, (ranked if kind == 'top' else shuffled)[:n])
                for kind, n in variants
            ]
        )
        with torch.no_grad():
            answers = question_scores(model, hidden)['probabilities'].argmax(1).cpu().numpy()
        for variant, answer in zip(variants, answers):
            flipped[variant].append(answer == UNFINISHED_IDX)
        by_dialect.setdefault(clip.label, []).append(flipped[('top', 3)][-1])

    curve = {
        kind: [float(np.mean(flipped[(kind, n)])) for n in hidden_counts]
        for kind in ('top', 'random')
    }
    dialect_share = {name: float(np.mean(values)) for name, values in sorted(by_dialect.items())}
    with open(f'{WS}/results/deletion_curve_{FINAL}.json', 'w') as file:
        json.dump(
            {'hidden': hidden_counts, **curve, 'by_dialect_3_top': dialect_share}, file, indent=1
        )

    fig, ax = plt.subplots(figsize=(HALF, 2.2))
    ax.plot(
        hidden_counts,
        curve['top'],
        marker='o',
        ms=3.5,
        color='#c0392b',
        label='nejdůležitější políčka',
    )
    ax.plot(
        hidden_counts, curve['random'], marker='s', ms=3.5, color='#7f8c8d', label='náhodná políčka'
    )
    ax.set_xlabel('počet zakrytých políček')
    ax.set_ylabel('podíl „nedozpívaný“')
    ax.set_ylim(0, 1)
    comma_ticks(ax, 'y', 1, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(hidden_counts)
    ax.legend(frameon=False, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, out, 'odstraneni_krivka')

    fig, ax = plt.subplots(figsize=(HALF, 2.2))
    names = [name for name in DIALECTS if name in dialect_share]
    ax.bar(names, [dialect_share[name] for name in names], color=[COLOURS[name] for name in names])
    ax.set_ylim(0, 1)
    comma_ticks(ax, 'y', 1, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel('podíl „nedozpívaný“')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, out, 'odstraneni_nareci')
    print(
        f'deletion curve: top {np.round(curve["top"], 3)}, random {np.round(curve["random"], 3)}, by dialect {dialect_share}',
        file=sys.stderr,
    )


def comma_ticks(ax, axis='y', digits=1, ticks=None):
    """Tick labels with a decimal comma, as Czech text writes numbers."""
    get = ax.get_yticks if axis == 'y' else ax.get_xticks
    put = ax.set_yticks if axis == 'y' else ax.set_xticks
    ticks = list(get()) if ticks is None else list(ticks)
    put(ticks, [f'{tick:.{digits}f}'.replace('.', ',') for tick in ticks])


#: The recipes of the final comparison, as the thesis names them, top to bottom.
FINAL_SERIES = [
    ('final_reference', 'výchozí, 0,90 M'),
    ('final_combined', 'kombinovaný, 0,90 M'),
    ('final_combined_430k', 'kombinovaný, 0,43 M'),
    ('final_combined_190k', 'kombinovaný, 0,19 M'),
    ('final_combined_170k_half_resolution', 'kombinovaný, 0,17 M, pol. rozlišení'),
    ('final_field_ready', 'nasazený s oběma změnami, 0,90 M'),
]


@figure
def recepty_semena(out):
    """The final comparison: accuracy of the three trainings of every recipe (dots) and their mean
    (bar): `recepty_semena.svg`."""
    fig, ax = plt.subplots(figsize=(FULL * 0.8, 2.3))
    for row, (run, name) in enumerate(FINAL_SERIES):
        values = [
            100 * load_run_predictions(f'{run}{seed}').correct.mean() for seed in ('', '_s1', '_s2')
        ]
        y = len(FINAL_SERIES) - 1 - row
        ax.plot([np.mean(values)] * 2, [y - 0.28, y + 0.28], color='#333333', lw=1.6)
        ax.scatter(
            values, [y] * 3, s=16, color='#e2231a' if 'combined' in run else '#7f7f7f', zorder=3
        )
    ax.set_yticks(range(len(FINAL_SERIES)), [name for _, name in FINAL_SERIES][::-1])
    ax.set_xlim(95.6, 97.0)
    comma_ticks(ax, 'x', 1, np.arange(95.6, 97.01, 0.2))
    ax.set_xlabel('přesnost (%)')
    ax.grid(axis='x', color='#dddddd', lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, out, 'recepty_semena')


#: Clips of the span-rule figure: (file name part, class, rank among the clips with spans, by the
#: reference model's confidence). Picked by eye from the first ten of each class.
SPAN_RULE_EXAMPLES = [
    ('BC', 'BC', 1),
    ('BhBl', 'BhBl', 0),
    ('XB', 'XB', 0),
    ('nedozpivany', 'Unfinished', 0),
]


@figure
def pravidlo_useku(out):
    """How the element spans are built without annotation: the spectrogram as the model sees it,
    the sustained tones the rule finds (white lines at their pitch) and the elements assigned to
    them from the clip's label (bars above): `pravidlo_<name>.svg`."""
    from evaluation.element_sequences import TOKEN_COLOURS
    from processing.build_element_spans_from_tones import (
        BIN_HZ,
        FRAME_SECONDS,
        Spectrograms,
        element_spans,
        labelled_window_start,
        tone_runs,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    spectrograms = Spectrograms(device)
    predictions = load_run_predictions(REFERENCE)
    bins = np.arange(len(BIN_HZ))
    for name, label, rank in SPAN_RULE_EXAMPLES:
        of_class = predictions[(predictions.label == label) & predictions.correct]
        found = 0
        for clip in of_class.sort_values(f'p_{label}', ascending=False).itertuples():
            picture = spectrograms(clip.npy, labelled_window_start(clip))
            spans = element_spans(picture, label)
            if spans is None or (label == 'Unfinished' and len(spans) < 2):
                continue
            if found == rank:
                break
            found += 1
        n_bins, n_frames = picture.shape
        strip = 1.17
        fig, ax = plt.subplots(figsize=(HALF, 1.75))
        ax.imshow(
            picture,
            origin='lower',
            aspect='auto',
            extent=[0, n_frames * FRAME_SECONDS, 0, n_bins],
            cmap='magma',
            vmin=0,
            vmax=2,
        )
        for run in tone_runs(picture):
            first_hz, last_hz = run['hz'] - run['slope'] / 2, run['hz'] + run['slope'] / 2
            ax.plot(
                [run['start'], run['end']],
                [np.interp(first_hz, BIN_HZ, bins), np.interp(last_hz, BIN_HZ, bins)],
                color='white',
                lw=1.0,
            )
        shown = [element for element, _, _ in spans]
        for (element, start, end), shown_name in zip(spans, shown):
            ax.axvspan(start, end, ymin=1 / strip, ymax=1.0, color=TOKEN_COLOURS[element])
            if end - start >= 0.12:
                ax.text(
                    (start + end) / 2,
                    n_bins * (1 + strip) / 2,
                    TOKEN_LETTER.get(shown_name, shown_name),
                    ha='center',
                    va='center',
                    fontsize=7.5,
                )
        ax.set_ylim(0, n_bins * strip)
        ticks = [np.interp(hz, BIN_HZ, bins) for hz in (2000, 4000, 8000)]
        ax.set_yticks(ticks, ['2', '4', '8'])
        ax.set_ylabel('kHz')
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlabel('čas (s)')
        fig.tight_layout(pad=0.4)
        save(fig, out, f'pravidlo_{name}')
        print(f'span rule {name}: {clip.name} {" ".join(shown)}', file=sys.stderr)


@figure
def spolehlivost(out):
    """Is the model's confidence honest? Accuracy of the answers by confidence
    (`spolehlivost_presnost.svg`) and the share of answers in every confidence band
    (`spolehlivost_podil.svg`), for the reported model."""
    predictions = load_run_predictions(FINAL)
    probabilities = predictions[[f'p_{name}' for name in CLASSES]].values
    confidence = probabilities.max(1)
    edges = np.array([0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    band = np.digitize(confidence, edges) - 1
    centres = [(edges[i] + min(edges[i + 1], 1.0)) / 2 for i in range(len(edges) - 1)]
    widths = [min(edges[i + 1], 1.0) - edges[i] for i in range(len(edges) - 1)]
    accuracy = [
        predictions.correct[band == i].mean() if (band == i).any() else np.nan
        for i in range(len(centres))
    ]
    share = [(band == i).mean() for i in range(len(centres))]

    fig, ax = plt.subplots(figsize=(HALF, 2.2))
    ax.bar(centres, accuracy, width=[w * 0.92 for w in widths], color='#19268e', alpha=0.85)
    ax.plot([0.3, 1.0], [0.3, 1.0], color='#7f7f7f', lw=1, ls='--')
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.3, 1.02)
    comma_ticks(ax, 'x', 1, [0.3, 0.5, 0.7, 0.9])
    comma_ticks(ax, 'y', 1, [0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('jistota modelu')
    ax.set_ylabel('podíl správných')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, out, 'spolehlivost_presnost')

    fig, ax = plt.subplots(figsize=(HALF, 2.2))
    ax.bar(centres, share, width=[w * 0.92 for w in widths], color='#9a9a9a')
    ax.set_xlim(0.3, 1.0)
    comma_ticks(ax, 'x', 1, [0.3, 0.5, 0.7, 0.9])
    ax.set_ylim(0, 1)
    comma_ticks(ax, 'y', 1, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('jistota modelu')
    ax.set_ylabel('podíl odpovědí')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, out, 'spolehlivost_podil')
    print(
        f'reliability: accuracy {np.round(accuracy, 3)}, share {np.round(share, 3)}',
        file=sys.stderr,
    )


@figure
def vyvazeni(out):
    """Class balancing in the sampler: each class's share of the labelled clips and of the
    examples drawn from them (chance of a clip proportional to 1 / sqrt(size of its class)):
    `vyvazeni.svg`."""
    labels = read_table('data/clips_audited.csv').label
    counts = labels.value_counts()
    order = ['BC', 'BE', 'BhBl', 'BlBh', 'XB', 'Unfinished', 'None']
    share_clips = np.array([counts[name] for name in order], float)
    share_clips /= share_clips.sum()
    share_draws = np.sqrt(np.array([counts[name] for name in order], float))
    share_draws /= share_draws.sum()
    positions = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(FULL * 0.8, 2.2))
    ax.bar(positions - 0.2, 100 * share_clips, 0.4, color='#9a9a9a', label='označené výřezy')
    ax.bar(positions + 0.2, 100 * share_draws, 0.4, color='#19268e', label='losované příklady')
    ax.set_xticks(positions, [CZECH[name] for name in order])
    ax.set_ylabel('podíl (%)')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, out, 'vyvazeni')
    print(
        'balance:',
        dict(zip(order, np.round(100 * share_clips, 1))),
        dict(zip(order, np.round(100 * share_draws, 1))),
        file=sys.stderr,
    )


#: The detector drawn in front of the classifier (trained on the songs the classifier's scan found).
DETECTOR_RUN = 'detector_band_tcn_scan_labels'


@figure
def detektor_nahravka(out):
    """The 8-thousand-parameter detector on the stretch of the voting figure: the spectrogram
    with the annotated songs, and the detector's song probability per window with the windows it
    passes to the classifier shaded (`detektor_nahravka.svg`)."""
    from config import effective_run_name
    from evaluation.whole_recording import scan_recording
    from training.checkpoint import load_model, run_checkpoints

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    songs = read_table('data/vizina_eval.csv')
    songs = songs[songs.split == 'test']
    npy, start, length, inside = voting_stretch(songs)
    models = [
        load_model(path).to(device)
        for path in run_checkpoints(f'{WS}/runs/{effective_run_name(DETECTOR_RUN)}')
    ]
    hop = 0.4
    starts, probabilities, _ = scan_recording(models, npy, hop, device)
    p_song = 1 - probabilities[:, NONE_IDX]
    fires = p_song >= 0.5
    passed = np.convolve(fires, np.ones(3), mode='same') > 0
    centres = starts + 2.0
    keep = (centres >= start) & (centres <= start + length)

    audio = read_audio(npy, start, start + length)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(FULL, 3.3), sharex=True, gridspec_kw={'height_ratios': [2, 1]}
    )
    draw_spectrogram(top, audio, '', band_khz=(2.5, 9.5))
    top.set_title('')
    for song in inside.itertuples():
        top.plot(
            [song.t0 - start, song.t1 - start],
            [9.15, 9.15],
            color=COLOURS[song.label],
            lw=3,
            solid_capstyle='butt',
        )
    top.set_ylabel('kHz')
    top.set_yticks([3, 5, 7, 9])
    for centre, is_passed in zip(centres[keep] - start, passed[keep]):
        if is_passed:
            bottom.axvspan(centre - hop / 2, centre + hop / 2, color='#d9e4f2', lw=0)
    bottom.plot(centres[keep] - start, p_song[keep], color='#19268e', lw=1.2)
    bottom.axhline(0.5, color='#7f7f7f', lw=0.8, ls='--')
    bottom.set_ylim(0, 1.05)
    comma_ticks(bottom, 'y', 1, [0, 0.5, 1.0])
    bottom.set_xlim(0, length)
    bottom.set_xticks(np.arange(0, length + 0.01, 2))
    bottom.set_xlabel('střed okna (s)')
    bottom.set_ylabel('p(zpěv)')
    bottom.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4, h_pad=1.0)
    save(fig, out, 'detektor_nahravka')
    print(
        f'detector: {passed[keep].mean():.2f} of the windows passed on this stretch',
        file=sys.stderr,
    )


#: Two trainings with fixed class directions (different random starts) for the fixed-map figure.
FIXED_MAP_RUNS = ['embedding_fixed_prototypes', 'embedding_fixed_prototypes_s1']


@figure
def pevna_mapa(out):
    """The fixed read-out map (`losses.fixed_prototypes.map_coordinates`) of the held-out clips of
    fold 0 for two trainings with fixed class directions that started from different random
    weights: `pevna_mapa_prvni.svg`, `pevna_mapa_druhe.svg`. Crosses are the class anchors."""
    from evaluation.embedding_map import held_out_embeddings
    from losses.fixed_prototypes import map_coordinates

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    first, clips = held_out_embeddings(FIXED_MAP_RUNS[0], 0, device)
    labels = np.asarray(clips.label, dtype=str)
    second = embed_clips(FIXED_MAP_RUNS[1], 0, clips, device)
    # the clips lie well inside the anchor circle: show where they are, not the anchors
    limits = ((-0.8, 0.8), (-1.0, 0.8))
    map_panel(out, 'pevna_mapa_prvni', map_coordinates(first), labels, limits=limits)
    map_panel(out, 'pevna_mapa_druhe', map_coordinates(second), labels, limits=limits)
    distance = np.linalg.norm(map_coordinates(first) - map_coordinates(second), axis=1)
    print(
        f'fixed map: {len(labels)} clips, median distance between trainings {np.median(distance):.3f}',
        file=sys.stderr,
    )


@figure
def vyska_hvizdu(out):
    """How high are the whistles? For every labelled clip where the span rule found the elements,
    the median pitch of the loudest band inside the B span, and inside the second whistle of BhBl
    and BlBh: `vyska_B.svg` (B by class) and `vyska_dvojice.svg` (first and second whistle of every
    BhBl and BlBh song, joined by a line). Numbers go to results/whistle_pitch.json."""
    from processing.build_element_spans_from_tones import (
        BIN_HZ,
        FRAME_SECONDS,
        Spectrograms,
        element_spans,
        labelled_window_start,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    spectrograms = Spectrograms(device)
    in_band = (BIN_HZ >= 2300) & (BIN_HZ <= 9800)
    band_hz = BIN_HZ[in_band] / 1000

    def pitch(picture, start, end):
        first = int(round(start / FRAME_SECONDS))
        last = max(int(round(end / FRAME_SECONDS)), first + 1)
        return float(np.median(band_hz[picture[in_band][:, first:last].argmax(0)]))

    rows = []
    for clip in read_table('data/clips_audited.csv').itertuples():
        picture = spectrograms(clip.npy, labelled_window_start(clip))
        spans = element_spans(picture, clip.label) or []
        row = {'label': clip.label}
        for element, start, end in spans:
            if element == 'B' and 'B' not in row:
                row['B'] = pitch(picture, start, end)
            if element in ('L', 'H') and 'second' not in row:
                row['second'] = pitch(picture, start, end)
        if 'B' in row:
            rows.append(row)
    table = pd.DataFrame(rows)
    order = ['BC', 'BE', 'XB', 'Unfinished', 'BhBl', 'BlBh']
    summary = {
        name: {
            'clips': int((table.label == name).sum()),
            'B_median_khz': float(table.B[table.label == name].median()),
            'second_median_khz': float(table.second[table.label == name].median())
            if name in ('BhBl', 'BlBh')
            else None,
        }
        for name in order
    }
    with open(f'{WS}/results/whistle_pitch.json', 'w') as file:
        json.dump(summary, file, indent=1)

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for position, name in enumerate(order):
        values = table.B[table.label == name].values
        if len(values) > 250:
            values = rng.choice(values, 250, replace=False)
        ax.scatter(
            position + rng.uniform(-0.28, 0.28, len(values)),
            values,
            s=2.5,
            color=COLOURS[name],
            alpha=0.7,
            linewidths=0,
        )
        median = table.B[table.label == name].median()
        ax.plot([position - 0.34, position + 0.34], [median] * 2, color='k', lw=1.4)
    ax.set_xticks(range(len(order)), [CZECH[name] for name in order], rotation=30, ha='right')
    ax.set_ylim(3.0, 7.5)
    comma_ticks(ax, 'y', 1, [3, 4, 5, 6, 7])
    ax.set_ylabel('výška hvizdu B (kHz)')
    ax.grid(axis='y', color='#e3e3e3', lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, out, 'vyska_B')

    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    for name in ('BhBl', 'BlBh'):
        pairs = table[(table.label == name) & table.second.notna()]
        for first, second in zip(pairs.B, pairs.second):
            ax.plot([0, 1], [first, second], color=COLOURS[name], alpha=0.5, lw=0.8)
    ax.set_xticks([0, 1], ['první hvizd (B)', 'druhý hvizd'])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(3.0, 7.5)
    comma_ticks(ax, 'y', 1, [3, 4, 5, 6, 7])
    ax.set_ylabel('výška (kHz)')
    ax.grid(axis='y', color='#e3e3e3', lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4)
    save(fig, out, 'vyska_dvojice')
    print(f'whistle pitch: {summary}', file=sys.stderr)


#: Clips that all 18 runs of the final comparison classify wrongly, one per group of the error
#: table in the thesis, picked by eye from the 32 such clips: (file name part, clip).
ERROR_EXAMPLES = [
    ('bez_X', 'F003154.wav.1.wav'),  # phrase, pause, low B, no X; labelled Unfinished -> XB
    ('jiny_druh', 'F001136.wav.7.wav'),  # phrase and B, then another species -> BE
    ('tvar', 'F001115.wav.1.wav'),  # BC with a short, flat last element -> BlBh
    ('slaby_hvizd', 'F001495.wav.10.wav'),  # BlBh with a faint second whistle -> Unfinished
    ('utrzky', 'F001639.wav.8.wav'),  # faint tonal fragments labelled None -> Unfinished
    ('zaver_na_zacatku', 'F001805.wav.10.wav'),  # only the end of a song at the start -> None
]


@figure
def trvale_chyby(out):
    """Clips every model of the final comparison gets wrong: `chyba_<name>.svg`, each with one
    second of its source recording on both sides; dashed lines mark the labelled window."""
    predictions = load_run_predictions('final_combined').set_index('name')
    for name, clip_name in ERROR_EXAMPLES:
        clip = predictions.loc[clip_name]
        start = labelled_window(clip)
        audio = read_audio(clip.npy, start - 1.0, start + 5.0)
        spectrogram_panel(out, f'chyba_{name}', audio, marks=(1.0, 5.0), height=1.6)
        print(f'error example {name}: {clip.label} -> {clip.predicted}', file=sys.stderr)


#: The BC song of the noise figure, picked by eye among the 40 most confident BC clips: its
#: ending is 10 dB quieter than its phrase, and its C element is clear in the original.
NOISE_EXAMPLE = 'F002136.wav.23.wav'


@figure
def sum_k_zakonceni(out):
    """What "signal-to-noise ratio" refers to: one BC song with the same stretch of steady
    background noise at the lowest level each recipe uses, 0 dB measured against the loud phrase
    ('band', the reference recipe) and 6 dB measured against the end of the song ('ending', the
    combined recipe): `sum_puvodni.svg`, `sum_k_frazi.svg`, `sum_k_zakonceni.svg`."""
    from augment.noise_mix import mix_noise

    predictions = load_run_predictions(REFERENCE)
    clip = predictions[predictions.name == NOISE_EXAMPLE].iloc[0]
    start = labelled_window(clip)
    audio = read_audio(clip.npy, start, start + 4.0)
    noise_bank = np.load(f'{WS}/data/noise_bank_clean.npy', allow_pickle=True)

    def steadiness(chunk):
        """Spread of the frame power in dB: low for steady noise such as wind or a stream."""
        chunk = np.asarray(chunk, np.float32)[: len(audio)]
        frames = chunk[: len(chunk) // 640 * 640].reshape(-1, 640)
        return float(np.std(10 * np.log10((frames**2).mean(1) + 1e-12)))

    noise = np.asarray(min(noise_bank, key=steadiness), np.float32)[: len(audio)]
    spectrogram_panel(out, 'sum_puvodni', audio, width=THIRD, height=1.6)
    spectrogram_panel(
        out, 'sum_k_frazi', mix_noise(audio, noise, 0.0, 'band'), width=THIRD, height=1.6
    )
    spectrogram_panel(
        out, 'sum_k_zakonceni', mix_noise(audio, noise, 6.0, 'ending'), width=THIRD, height=1.6
    )


KONTRAFAKT_RANK = {'XB': 12, 'BC': 2}


@figure
def kontrafakt(out):
    """Counterfactual pairs for an XB and a BC song: `kontrafakt_<class>_puvodni.svg`,
    `..._bez_vysek.svg` (nothing above 6 kHz), `..._bez_zakonceni.svg` (ending erased)."""
    from augment.treble_cut import treble_cut

    erased = read_table('data/erased_endings_v2.csv')
    clips = read_table('data/clips_audited.csv').set_index('name')
    for original_class in ('XB', 'BC'):
        candidates = erased[erased.was == original_class]
        candidates = candidates[candidates.name.str.split('erased:').str[1].isin(clips.index)]
        candidates = candidates.sort_values('p_unfinished', ascending=False)
        pair = candidates.iloc[KONTRAFAKT_RANK[original_class]]
        original = clips.loc[pair['name'].split('erased:')[1]]
        start = labelled_window(original)
        audio = read_audio(original.npy, start, start + 4.0)
        versions = {
            'puvodni': audio,
            'bez_vysek': with_floor(treble_cut(audio, 6000.0)),
            'bez_zakonceni': read_audio(pair.npy, 0.0, 4.0),
        }
        for name, version in versions.items():
            spectrogram_panel(out, f'kontrafakt_{original_class}_{name}', version)


# --------------------------------------------------------------------------- element sequences
#: The run whose element-token output is drawn.
SEQUENCE_RUN = os.environ.get('THESIS_SEQUENCE_RUN', 'element_sequence_ctc')
#: Letter drawn at a decoded token ("F" stands for the phrase, "fráze").
TOKEN_LETTER = {'phrase': 'F'}
#: The clips shown: (file name part, class, rank by confidence among the correctly classified
#: clips of the class). Picked by eye from the first twelve of each class: four songs whose
#: tokens land on their elements, and two that show the limits (a phrase-only song decoded with a
#: B it does not have; a window with the end of one song and the start of the next).
SEQUENCE_EXAMPLES = [
    ('XB', 'XB', 1),
    ('BC', 'BC', 6),
    ('BhBl', 'BhBl', 10),
    ('nedozpivany_s_B', 'Unfinished', 9),
    ('nedozpivany_bez_B', 'Unfinished', 0),
    ('dva_zpevy', 'BC', 5),
]


def sequence_picture(out, name, model, clip, device):
    """One clip. Top: its spectrogram with the decoded tokens as coloured boxes in a strip above
    it, each at the time the token was emitted. Bottom: the probability of every element token
    in every 40 ms frame (what the CTC output looks like). Returns the decoded token names."""
    from evaluation.element_sequences import TOKEN_COLOURS
    from losses.element_ctc import TOKENS, decode_tokens

    spectrogram = clip_spectrogram(model, clip, device)
    with torch.no_grad():
        token_log_probs = model(spectrogram[None])['token_log_probs']
    probabilities = token_log_probs[0].exp().cpu().numpy()  # (frames, tokens), token 0 = blank
    sequences, spans = decode_tokens(token_log_probs)
    picture = spectrogram[0].cpu().numpy()
    n_bins = picture.shape[0]
    frame_seconds = 4.0 / len(probabilities)
    time = (np.arange(len(probabilities)) + 0.5) * frame_seconds
    strip = 1.17  # the axis is this much taller than the spectrogram: room for the token strip

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(HALF, 2.35), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]}
    )
    top.imshow(
        picture,
        origin='lower',
        aspect='auto',
        extent=[0, 4.0, 0, n_bins],
        cmap='magma',
        vmin=0,
        vmax=2,
    )
    top.set_ylim(0, n_bins * strip)
    # the whistles keep the network's names: B, then L (lower) or H (higher) than the B before
    shown_names = [token for token, _, _ in spans[0]]
    box_centre = -1.0
    for (token, first_frame, last_frame), shown in zip(spans[0], shown_names):
        emitted = (first_frame + last_frame + 1) / 2 * frame_seconds
        # boxes of tokens emitted in neighbouring frames are drawn side by side, not on top of
        # each other, and the first and last stay inside the picture
        box_centre = min(max(emitted, box_centre + 0.26, 0.13), 3.87)
        top.axvspan(
            box_centre - 0.13,
            box_centre + 0.13,
            ymin=1 / strip,
            ymax=1.0,
            color=TOKEN_COLOURS[token],
        )
        top.text(
            box_centre,
            n_bins * (1 + strip) / 2,
            TOKEN_LETTER.get(shown, shown),
            ha='center',
            va='center',
            fontsize=7.5,
        )
    ticks, tick_labels = frequency_ticks(
        (model.frontend.bank.fmin, model.frontend.bank.fmax), n_bins
    )
    keep = [index for index, label in enumerate(tick_labels) if label in ('2', '4', '8')]
    top.set_yticks([ticks[index] for index in keep], [tick_labels[index] for index in keep])
    top.set_ylabel('kHz')

    for index, token in enumerate(TOKENS):
        if index:
            bottom.plot(time, probabilities[:, index], color=TOKEN_COLOURS[token], lw=1.1)
    bottom.set_ylim(0, 1.1)
    bottom.set_yticks([0, 1])
    bottom.set_xlim(0, 4.0)
    bottom.set_xticks([0, 1, 2, 3, 4])
    bottom.set_xlabel('čas (s)')
    bottom.set_ylabel('pravděp.')
    bottom.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4, h_pad=0.6)
    save(fig, out, name)
    return shown_names


@figure
def sekvence_elementu(out):
    """The element-token output on six held-out clips: `sekvence_<name>.svg`. The decoded
    sequences are printed, for the captions."""
    predictions = load_run_predictions(SEQUENCE_RUN)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_models = FoldModels(SEQUENCE_RUN, device)
    for name, label, rank in SEQUENCE_EXAMPLES:
        of_class = predictions[(predictions.label == label) & predictions.correct]
        clip = of_class.sort_values(f'p_{label}', ascending=False).iloc[rank]
        model = fold_models.get(clip.cv_fold)
        tokens = sequence_picture(out, f'sekvence_{name}', model, clip, device)
        print(f'sequence of {name} ({clip.name}): {" ".join(tokens)}', file=sys.stderr)


# ------------------------------------------------------------------------ per-frame element tags
#: The run whose per-frame element tags are drawn (`element_frames=True`, trained on spans found
#: from label and spectrogram, `processing/build_element_spans_from_tones.py`).
FRAMES_RUN = os.environ.get('THESIS_FRAMES_RUN', 'element_frames_from_tones')
#: The clips shown: (file name part, class, which of the correctly classified clips whose tagged
#: elements give the right class, by confidence). Picked by eye like SEQUENCE_EXAMPLES, as
#: counterparts of the CTC figure: four complete songs, an unfinished song ending in B, and one
#: that ends with the phrase alone (the case the CTC output reads as "phrase, B").
FRAMES_EXAMPLES = [
    ('XB', 'XB', 1),
    ('BC', 'BC', 6),
    ('BhBl', 'BhBl', 2),
    ('BlBh', 'BlBh', 2),
    ('nedozpivany_s_B', 'Unfinished', 9),
    ('nedozpivany_bez_B', 'Unfinished', 0),
]


def element_runs(model, spectrogram):
    """The tagged elements of one window [(name, start_s, end_s), ...] and the per-frame tag
    probabilities (frames, tags)."""
    from losses.element_ctc import FRAME_TAGS, decode_frame_tags, refine_boundaries

    with torch.no_grad():
        logits = model(spectrogram[None])['element_logits'][0].float()  # (tags, frames)
    probabilities = logits.softmax(0).T.cpu().numpy()
    frame_seconds = 4.0 / probabilities.shape[0]
    runs = refine_boundaries(probabilities, decode_frame_tags(probabilities.argmax(1)))
    runs = [(name, first * frame_seconds, end * frame_seconds) for name, first, end in runs]
    assert len(FRAME_TAGS) == probabilities.shape[1]
    return runs, probabilities


def frames_picture(out, name, model, clip, device):
    """One clip. Top: the spectrogram with every tagged element as a bar over its whole length in a
    strip above it. Bottom: the probability of every element tag in every 40 ms frame. Returns the
    element names as shown."""
    from evaluation.element_sequences import TOKEN_COLOURS
    from losses.element_ctc import FRAME_TAGS

    spectrogram = clip_spectrogram(model, clip, device)
    runs, probabilities = element_runs(model, spectrogram)
    shown_names = [element for element, _, _ in runs]
    picture = spectrogram[0].cpu().numpy()
    n_bins = picture.shape[0]
    frame_seconds = 4.0 / probabilities.shape[0]
    time = (np.arange(probabilities.shape[0]) + 0.5) * frame_seconds
    strip = 1.17

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(HALF, 2.35), sharex=True, gridspec_kw={'height_ratios': [2.1, 1]}
    )
    top.imshow(
        picture,
        origin='lower',
        aspect='auto',
        extent=[0, 4.0, 0, n_bins],
        cmap='magma',
        vmin=0,
        vmax=2,
    )
    top.set_ylim(0, n_bins * strip)
    for (element, start, end), shown in zip(runs, shown_names):
        top.axvspan(start, end, ymin=1 / strip, ymax=1.0, color=TOKEN_COLOURS[element])
        if end - start >= 0.12:
            top.text(
                (start + end) / 2,
                n_bins * (1 + strip) / 2,
                TOKEN_LETTER.get(shown, shown),
                ha='center',
                va='center',
                fontsize=7.5,
            )
    ticks, tick_labels = frequency_ticks(
        (model.frontend.bank.fmin, model.frontend.bank.fmax), n_bins
    )
    keep = [index for index, label in enumerate(tick_labels) if label in ('2', '4', '8')]
    top.set_yticks([ticks[index] for index in keep], [tick_labels[index] for index in keep])
    top.set_ylabel('kHz')

    for index, tag in enumerate(FRAME_TAGS):
        if index and probabilities[:, index].max() > 0.05:
            bottom.plot(time, probabilities[:, index], color=TOKEN_COLOURS[tag], lw=1.1)
    bottom.set_ylim(0, 1.1)
    bottom.set_yticks([0, 1])
    bottom.set_xlim(0, 4.0)
    bottom.set_xticks([0, 1, 2, 3, 4])
    bottom.set_xlabel('čas (s)')
    bottom.set_ylabel('pravděp.')
    bottom.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(pad=0.4, h_pad=0.6)
    save(fig, out, name)
    return shown_names


@figure
def useky_prvku(out):
    """The per-frame element tags on six held-out clips: `useky_<name>.svg`. The tagged elements
    are printed, for the captions."""
    from losses.element_ctc import STRNADI_GRAMMAR, canonical_elements

    predictions = load_run_predictions(FRAMES_RUN)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fold_models = FoldModels(FRAMES_RUN, device)
    for name, label, rank in FRAMES_EXAMPLES:
        of_class = predictions[(predictions.label == label) & predictions.correct]
        found = 0
        for clip in of_class.sort_values(f'p_{label}', ascending=False).itertuples():
            model = fold_models.get(clip.cv_fold)
            runs, _ = element_runs(model, clip_spectrogram(model, clip, device))
            names = canonical_elements([element for element, _, _ in runs])
            if STRNADI_GRAMMAR.class_of_sequence(names) != label:
                continue
            if name == 'nedozpivany_bez_B' and names != ['phrase']:
                continue
            if name == 'nedozpivany_s_B' and names != ['phrase', 'B']:
                continue
            if found == rank:
                shown = frames_picture(out, f'useky_{name}', model, clip, device)
                print(f'elements of {name} ({clip.name}): {" ".join(shown)}', file=sys.stderr)
                break
            found += 1


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
