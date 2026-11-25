"""The appendix table of the written thesis: every experiment with its result (Czech).

    python -m tools.thesis_tables <thesis directory>

Writes `prilohy/02-prehled-pokusu.typ` and `kapitoly/ml-matice-zamen.typ` (Typst).

The appendix: every finished 5-fold run scored on the audited clips gets one row: a description in words, parameters, accuracy, macro-F1, the lowest class
recall and, when `tools/model_scorecard.py` has been run, the time of one window on one CPU
thread. Runs are described, never named: the thesis text carries no run identifiers.
"""

import json
import os
import sys

import pandas as pd

from config import CLASSES, WS
from evaluation.report import load_run_predictions
from tools.results_table import describe_run

CZECH_CLASS = {'Unfinished': 'nedozp.', 'None': 'bez strnada'}

#: (group heading, [(run, description), ...]) in the order of the appendix.
GROUPS = [
    (
        'Výchozí recept',
        [
            ('strnadnet_audited_labels', 'výchozí recept, první semínko'),
            ('strnadnet_audited_labels_s1', 'výchozí recept, druhé semínko'),
            ('strnadnet_audited_labels_s2', 'výchozí recept, třetí semínko'),
            ('strnadnet_field_ready', 'nasazený recept'),
            ('strnadnet_field_ready_190k', 'nasazený recept s nejmenší sítí'),
        ],
    ),
    (
        'Srovnání hlavních receptů (výsledná verze označení, tři semínka)',
        [
            (f'{run}{seed}', f'{description}, {order} semínko')
            for run, description in [
                ('final_reference', 'výchozí recept'),
                ('final_combined', 'kombinovaný recept'),
                ('final_combined_430k', 'kombinovaný recept, síť 0,43 M'),
                ('final_combined_190k', 'kombinovaný recept, síť 0,19 M'),
                ('final_combined_170k_half_resolution', 'kombinovaný recept, poloviční rozlišení'),
                ('final_field_ready', 'nasazený recept s oběma změnami'),
            ]
            for seed, order in [('', 'první'), ('_s1', 'druhé'), ('_s2', 'třetí')]
        ]
        + [
            ('final_combined_tuned', 'kombinovaný recept s nastavením z ladění'),
            ('final_combined_40_epochs', 'kombinovaný recept, 40 epoch'),
            ('tuned_reference', 'výchozí recept s nastavením z ladění'),
        ],
    ),
    (
        'Velikost a rychlost vlastní sítě',
        [
            ('size_strnadnet_190k', 'nejmenší síť'),
            ('size_strnadnet_430k', 'menší síť'),
            ('size_strnadnet_2900k', 'větší síť'),
            ('speed_half_resolution', 'poloviční rozlišení spektrogramu'),
            ('speed_depthwise', 'hloubkově separabilní konvoluce'),
            ('golf_190k_half_resolution', 'nejmenší síť s polovičním rozlišením'),
            ('golf_430k_half_resolution', 'menší síť s polovičním rozlišením'),
        ],
    ),
    (
        'Jiné architektury',
        [
            ('temporal_gru', 'rekurentní vrstva místo dilatovaných konvolucí'),
            ('temporal_attention', 'self-attention místo dilatovaných konvolucí'),
            ('ending_zoom', 'podrobný pohled na závěr zpěvu nalezený podle energie'),
            ('zoom_on_attention', 'podrobný pohled na místo s vrcholem pozornosti'),
            ('strnadnet_tonal', 'úvodní vrstva s dlouhými filtry pro tóny'),
            ('fdy_crnn', 'konvolučně-rekurentní síť s frekvenčně dynamickými konvolucemi'),
            ('bc_resnet_tiny', 'BC-ResNet'),
            ('contour_tcn', 'jednorozměrná síť nad konturou vrcholové frekvence'),
            ('patch_transformer', 'transformer nad výřezy spektrogramu'),
            ('backbone_mobilenetv3_large_imagenet', 'MobileNetV3 předtrénovaná na obrazech'),
            ('backbone_efficientnet_b0_imagenet', 'EfficientNet-B0 předtrénovaná na obrazech'),
            ('backbone_convnext_atto_imagenet', 'ConvNeXt-Atto předtrénovaná na obrazech'),
            ('backbone_regnety_imagenet', 'RegNetY předtrénovaná na obrazech'),
            ('backbone_densenet121_imagenet', 'DenseNet121 předtrénovaná na obrazech'),
            ('legacy_densenet121', 'architektura původního modelu, nové učení'),
            ('legacy_mobilenetv3_delta', 'starší návrh s delta příznaky'),
            ('perch_frozen_temporal', 'zmrazený Perch s časovým modulem'),
        ],
    ),
    (
        'Vstupní reprezentace',
        [
            ('frontend_percentile_floor', 'odečtení nízkého percentilu místo mediánu'),
            ('frontend_pcen', 'normalizace PCEN'),
            ('frontend_linear_hires', 'lineární frekvenční osa s vyšším rozlišením'),
            ('frontend_multires', 'tři délky okna STFT současně'),
            ('frontend_delta_channels', 'kanály s časovou derivací (delta, delta-delta)'),
        ],
    ),
    (
        'Trénovací signál',
        [
            ('ablation_flat_head', 'plochá hlava se sedmi výstupy'),
            ('ablation_clip_labels_only', 'jen označené výřezy, bez slabých označení'),
            ('ablation_hard_negatives', 'těžké negativní příklady jiných druhů'),
            ('augment_real_world', 'simulace vzdálenosti, jiné druhy, různé mikrofony'),
            ('augment_background_songs', 'druhý, tišší strnad v pozadí'),
            ('augment_ending_referenced_noise', 'hlasitost šumu vztažená k zakončení zpěvu'),
            ('device_robustness', 'simulace kodeků a zařízení'),
            ('counterfactual_treble_cut', 'kontrafaktuální příklady: odříznutí výšek'),
            ('counterfactual_erased_endings', 'kontrafaktuální příklady: vymazané zakončení'),
            ('bag_completeness', 'úplnost učená z celých nahrávek (ztráta nad množinou zpěvů)'),
            ('loss_noise_robust', 'zobecněná křížová entropie'),
            ('loss_focal', 'fokální ztráta'),
            ('loss_class_weighted', 'váhy tříd ve ztrátě místo ve výběru'),
        ],
    ),
    (
        'Rozšiřování dat a posloupnosti prvků',
        [
            ('expanded_machine_labels', 'zpěvy označené modelem, první kolo'),
            ('expanded_machine_labels_round2', 'zpěvy označené modelem, druhé kolo'),
            ('expanded_scan_harvest', 'zpěvy nově nalezené průchodem nahrávek'),
            ('expanded_all_sources', 'všechny zdroje včetně pasivních rekordérů'),
            ('element_sequence_ctc', 'výstup posloupnosti prvků (CTC)'),
            ('element_sequence_discriminative', 'posloupnost prvků, třídy soutěží'),
            ('element_sequence_anchored', 'posloupnost prvků ukotvená časy od ornitologů'),
            ('element_sequence_ctc_bh_bl', 'posloupnost prvků s hvizdy pojmenovanými Bh a Bl'),
            ('element_frames_from_tones', 'úseky prvků, hvizdy pojmenované relativně (B, L, H)'),
            ('element_frames_bh_bl_tags', 'úseky prvků, hvizdy pojmenované Bh a Bl'),
        ],
    ),
    (
        'Menší množství označených dat',
        [
            ('learning_curve_10pct', '10 % označených výřezů'),
            ('learning_curve_25pct', '25 % označených výřezů'),
            ('learning_curve_50pct', '50 % označených výřezů'),
            ('learning_curve_10pct_with_weak_songs', '10 % označených výřezů a slabá označení'),
            ('learning_curve_25pct_with_weak_songs', '25 % označených výřezů a slabá označení'),
            ('learning_curve_50pct_with_weak_songs', '50 % označených výřezů a slabá označení'),
        ],
    ),
]


def number(value, digits):
    """A number with a decimal comma, for Typst text."""
    return f'{value:.{digits}f}'.replace('.', ',')


def table_rows():
    """Typst source of the table body, one line per finished run, and how many runs it has."""
    timed = {}
    scorecard = f'{WS}/results/model_scorecard.json'
    if os.path.exists(scorecard):
        with open(scorecard) as file:
            timed = {row['run']: row['window_ms'] for row in json.load(file)}

    lines, n_runs = [], 0
    for heading, runs in GROUPS:
        group_lines = []
        for run, description in runs:
            row = describe_run(f'{WS}/runs/{run}', 'data/clips_audited.csv')
            if row is None or row['folds'] != 5:
                continue
            weakest = min(CLASSES, key=lambda name: row['recall'][name])
            time = f'{number(timed[run], 0)} ms' if run in timed else '--'
            group_lines.append(
                f'  [{description}], [{number(row["parameters"] / 1e6, 2)} M], '
                f'[{number(100 * row["accuracy"], 1)} %], [{number(row["macro_f1"], 3)}], '
                f'[{CZECH_CLASS.get(weakest, weakest)} {number(100 * row["recall"][weakest], 0)} %], '
                f'[{time}],'
            )
        if group_lines:
            lines.append(f'  table.cell(colspan: 6)[*{heading}*],')
            lines += group_lines
            n_runs += len(group_lines)
    return '\n'.join(lines), n_runs


CLASS_ORDER = ['BC', 'BE', 'BhBl', 'BlBh', 'XB', 'Unfinished', 'None']
CZECH_CLASS_FULL = {'Unfinished': 'nedozpívaný', 'None': 'bez strnada'}
#: Column headings of the confusion matrix (the columns are narrow).
HEADER_CLASS = {'Unfinished': 'nedozp.', 'None': 'bez \\ strnada'}


def confusion_matrix_source(run):
    """Typst source of the out-of-fold confusion matrix of a run: counts, each cell shaded by its
    share of the row (white to dark blue), so the table stays editable text in the thesis."""
    predictions = load_run_predictions(run)
    counts = pd.crosstab(predictions.label, predictions.predicted)
    counts = counts.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0).values
    share = counts / counts.sum(1, keepdims=True)
    names = [CZECH_CLASS_FULL.get(name, name) for name in CLASS_ORDER]

    lines = [
        '// Generated by tools/thesis_tables.py of the classifier repository.',
        '#table(',
        '  columns: (auto,) + (1fr,) * 7,',
        '  stroke: 0.5pt,',
        '  align: center + horizon,',
        '  inset: (x: 4pt, y: 7pt),',
        '  table.cell(rowspan: 2, align: left + horizon)[*označení*],',
        '  table.cell(colspan: 7)[*odpověď modelu*],',
        '  ' + ', '.join(f'[{HEADER_CLASS.get(name, name)}]' for name in CLASS_ORDER) + ',',
    ]
    for row, name in enumerate(names):
        cells = [f'table.cell(align: left + horizon)[{name}]']
        for column in range(len(names)):
            if counts[row, column] == 0:
                cells.append('[]')
                continue
            # white (0) to the blue of the thesis figures (1)
            red, green, blue = (
                round(255 + (end - 255) * share[row, column]) for end in (31, 78, 156)
            )
            text = str(counts[row, column])
            if share[row, column] > 0.5:
                text = f'#text(fill: white)[{text}]'
            cells.append(f'table.cell(fill: rgb({red}, {green}, {blue}))[{text}]')
        lines.append('  ' + ', '.join(cells) + ',')
    lines.append(')')
    return '\n'.join(lines) + '\n'


def main():
    thesis = sys.argv[1]
    final_run = os.environ.get('THESIS_FINAL_RUN', 'strnadnet_field_ready')
    matrix_path = os.path.join(thesis, 'kapitoly', 'ml-matice-zamen.typ')
    with open(matrix_path, 'w', encoding='utf-8') as file:
        file.write(confusion_matrix_source(final_run))
    print(f'confusion matrix of {final_run} -> {matrix_path}')
    rows, n_runs = table_rows()
    source = f"""// Generated by tools/thesis_tables.py of the classifier repository; do not edit by hand.
= Přehled pokusů s klasifikátorem nářečí
<appendix:prehled-pokusu>

Tabulka shrnuje všechny pokusy vyhodnocené stejným způsobem: pětinásobnou křížovou validací po nahrávkách na revidovaných výřezech. Není-li uvedeno jinak, liší se řádek od výchozího receptu jedinou uvedenou změnou. Sloupec _nejslabší třída_ udává třídu s nejnižší citlivostí a tuto citlivost; čas je doba zpracování jednoho čtyřsekundového okna na jednom vlákně procesoru.

#[
#set text(size: 9pt)
#show figure: set block(breakable: true)
#figure(
  table(
    columns: (1fr, auto, auto, auto, auto, auto),
    stroke: 0.5pt,
    align: (left, right, right, right, left, right),
    table.header([*Pokus*], [*Parametry*], [*Přesnost*], [*Makro-F1*], [*Nejslabší třída*], [*Čas okna*]),
{rows}
  ),
  caption: [Výsledky všech {n_runs} pokusů vyhodnocených křížovou validací po nahrávkách],
) <tab:prehled-pokusu>
]
"""
    path = os.path.join(thesis, 'prilohy', '02-prehled-pokusu.typ')
    with open(path, 'w', encoding='utf-8') as file:
        file.write(source)
    print(f'{n_runs} runs -> {path}')


if __name__ == '__main__':
    main()
