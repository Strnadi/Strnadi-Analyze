"""Does a converted TFLite model give the same answers as the PyTorch model it came from?

    python -m tests.check_tflite <model.pt | run_dir> <file.tflite> [more.tflite ...] [--clips 400]

Needs the workspace (real clips), torch and ai-edge-litert. The clips are the out-of-fold clips of
the checkpoint's fold when it has stored predictions beside it, otherwise a sample of the labelled
table. Raw (not normalised) audio goes into both models: the exported graph normalises itself.

Reported per file: size, largest probability difference, share of clips with the same predicted
class, accuracy, and milliseconds per 4 s clip on one thread. A float16 or float32 file should
agree on every clip; anything else means the conversion is broken (see serving/export_tflite.py).
"""

import os
import sys
import time

import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter

from config import C2I
from serving.export_onnx import Deploy
from training.checkpoint import export_mode, load_model, run_checkpoints
from training.data import load_seg, read_table

CLIP_SAMPLES = 128000


def test_clips(checkpoint_path, n_clips):
    """Labelled clips to compare on, as (waveforms, labels)."""
    table = read_table('data/orig_index.csv')
    oof_path = os.path.join(os.path.dirname(checkpoint_path), 'oof.npz')
    if os.path.exists(oof_path):
        rows = table.loc[np.asarray(np.load(oof_path, allow_pickle=True)['idx'], int)]
    else:
        rows = table[~table.aug]
    rows = rows.sample(min(n_clips, len(rows)), random_state=0)
    waves = [load_seg(row.npy, row.t0, row.t0 + 4, CLIP_SAMPLES) for row in rows.itertuples()]
    return np.stack(waves).astype(np.float32), rows.label.map(C2I).values


def tflite_probabilities(path, waves):
    """Class probabilities (N, 7) and milliseconds per clip of one TFLite file."""
    interpreter = Interpreter(model_path=path, num_threads=1)
    input_detail = interpreter.get_input_details()[0]
    interpreter.resize_tensor_input(input_detail['index'], [1, CLIP_SAMPLES])
    interpreter.allocate_tensors()
    # The graph has two outputs, probabilities (.., 7) and the embedding.
    output_detail = [o for o in interpreter.get_output_details() if o['shape'][-1] == len(C2I)][0]

    probabilities = []
    started = time.time()
    for wave in waves:
        interpreter.set_tensor(input_detail['index'], wave[None])
        interpreter.invoke()
        probabilities.append(interpreter.get_tensor(output_detail['index'])[0].copy())
    milliseconds = 1000 * (time.time() - started) / len(waves)
    return np.array(probabilities), milliseconds


def main():
    arguments = sys.argv[1:]
    n_clips = 400
    if '--clips' in arguments:
        position = arguments.index('--clips')
        n_clips = int(arguments[position + 1])
        del arguments[position : position + 2]
    checkpoints = run_checkpoints(arguments[0])
    tflite_paths = arguments[1:]
    torch.set_num_threads(1)

    waves, labels = test_clips(checkpoints[0], n_clips)
    reference_model = Deploy([load_model(path) for path in checkpoints]).eval()
    export_mode(reference_model, True)
    with torch.no_grad():
        started = time.time()
        reference_model(torch.from_numpy(waves[:20]))
        torch_milliseconds = 1000 * (time.time() - started) / 20
        reference = reference_model(torch.from_numpy(waves))[0].numpy()
    reference_accuracy = float((reference.argmax(1) == labels).mean())
    print(
        f'{len(waves)} clips; PyTorch: accuracy {reference_accuracy:.4f}, {torch_milliseconds:.0f} ms / clip'
    )

    all_fine = True
    for path in tflite_paths:
        probabilities, milliseconds = tflite_probabilities(path, waves)
        largest_difference = float(np.abs(probabilities - reference).max())
        same_class = float((probabilities.argmax(1) == reference.argmax(1)).mean())
        accuracy = float((probabilities.argmax(1) == labels).mean())
        print(
            f'{os.path.basename(path):40s} {os.path.getsize(path) / 1e6:5.2f} MB  '
            f'max |dp| {largest_difference:.4f}  same class {same_class:.4f}  '
            f'accuracy {accuracy:.4f}  {milliseconds:.0f} ms / clip'
        )
        is_float = 'float' in os.path.basename(path)
        if is_float and same_class < 1.0:
            all_fine = False
    if not all_fine:
        raise SystemExit('a float TFLite file disagrees with PyTorch: the conversion is broken')
    print('float models agree with PyTorch on every clip')


if __name__ == '__main__':
    main()
