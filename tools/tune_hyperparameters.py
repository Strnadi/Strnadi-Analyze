"""Random search over the training hyper-parameters of the reference recipe.

    python -m tools.tune_hyperparameters [--trials 16] [--name tune_reference]

The learning rate, epochs, table shares and augmentation strengths of the reference were taken
over from the first research session and never searched. Each trial draws one value per entry of
`SEARCH_SPACE`, trains folds 0-2 with early stopping, and is scored on the held-out clips of those
three folds. Trial 0 is the reference itself, so every trial is compared under the same protocol.

**Folds 3 and 4 are never seen by the search.** The best trial is afterwards trained on all five
folds as an ordinary experiment; its score on folds 3 and 4 is the honest check of whether the
search found something real or fitted the noise of folds 0-2 (two seeds of one configuration
differ by about 0.005 accuracy).

Results: `$STRNADI_WS/results/<name>.csv`, one row per trial, best first when printed. Trials are
ordinary runs (`runs/<name>_<nn>/`, with TensorBoard logs); a finished trial is not repeated.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config import CLASSES, WS
from evaluation.report import load_run_predictions
from model.strnadnet import StrnadNet
from training.fit import cross_validate
from training.settings import TrainSettings

SEARCH_FOLDS = [0, 1, 2]

#: What is searched: setting -> candidate values. The first value of each is the reference's.
SEARCH_SPACE = {
    'learning_rate': [0.002, 0.001, 0.003, 0.005],
    'weight_decay': [0.05, 0.01, 0.1],
    'epochs': [25, 40],
    'batch_size': [48, 32, 96],
    'label_smoothing': [0.05, 0.0, 0.1],
    'ema_decay': [0.998, 0.995, 0.999],
    'class_balance_power': [0.5, 0.3, 0.7],
    'weak_share': [0.35, 0.2, 0.5],  # share of draws from the weakly labelled songs
    'noise_probability': [0.5, 0.3, 0.7],
    'snr_db_range': [(0.0, 20.0), (5.0, 25.0)],
    'speed_jitter': [0.02, 0.0, 0.04],
    'freq_mask_width': [8, 4, 12],
    'dropout': [0.2, 0.1, 0.3],
}


def draw_trial(trial):
    """The hyper-parameters of a trial: the reference for trial 0, a random draw otherwise."""
    if trial == 0:
        return {name: values[0] for name, values in SEARCH_SPACE.items()}
    rng = np.random.default_rng(trial)
    return {name: values[rng.integers(len(values))] for name, values in SEARCH_SPACE.items()}


def trial_settings(parameters):
    """`TrainSettings` of the reference recipe with a trial's values filled in."""
    return TrainSettings(
        labelled_table='data/clips_audited.csv',
        extra_tables={
            'data/weak_v5.csv': parameters['weak_share'],
            'data/cf_v4.csv': 0.08,
            'data/vizann_v1.csv': 0.07,
        },
        noise_bank='data/noise_bank_clean.npy',
        eval_window='center',
        folds=SEARCH_FOLDS,
        early_stopping_patience=8,
        samples_per_epoch=6144,
        epochs=parameters['epochs'],
        batch_size=parameters['batch_size'],
        learning_rate=parameters['learning_rate'],
        weight_decay=parameters['weight_decay'],
        label_smoothing=parameters['label_smoothing'],
        ema_decay=parameters['ema_decay'],
        class_balance_power=parameters['class_balance_power'],
        noise_probability=parameters['noise_probability'],
        snr_db_range=parameters['snr_db_range'],
        snr_reference='band',
        splice_probability=0.3,
        speed_jitter=parameters['speed_jitter'],
        freq_mask_width=parameters['freq_mask_width'],
    )


def score_trial(run):
    """Accuracy and macro-F1 on the held-out clips of the search folds."""
    predictions = load_run_predictions(run)
    labels = predictions.label.map(CLASSES.index).values
    predicted = predictions.predicted.map(CLASSES.index).values
    return float((predicted == labels).mean()), float(f1_score(labels, predicted, average='macro'))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--trials', type=int, default=16)
    parser.add_argument('--name', default='tune_reference')
    arguments = parser.parse_args()

    rows = []
    for trial in range(arguments.trials):
        parameters = draw_trial(trial)
        run = f'{arguments.name}_{trial:02d}'
        dropout = parameters['dropout']
        cross_validate(
            run, lambda: StrnadNet(head='factorized', drop=dropout), trial_settings(parameters)
        )
        accuracy, macro_f1 = score_trial(run)
        rows.append({'trial': trial, 'accuracy': accuracy, 'macro_f1': macro_f1, **parameters})
        pd.DataFrame(rows).to_csv(f'{WS}/results/{arguments.name}.csv', index=False)
        print(
            f'trial {trial}: accuracy {accuracy:.4f}  macro-F1 {macro_f1:.4f}  {parameters}',
            flush=True,
        )

    table = pd.DataFrame(rows).sort_values(['accuracy', 'macro_f1'], ascending=False)
    print('\nbest first (folds 0-2 only; confirm the winner on folds 3-4):')
    print(table.to_string(index=False))


if __name__ == '__main__':
    main()
