from . import birdnet_preprocess_interface
from . import wav_helper
from . import spectrogram

from typing import *
import numpy as np
import logging

logger = logging.getLogger("preprocess")

def preprocess(wav_bytes :bytes) -> List[Tuple[Tuple[float, float], np.ndarray[np.floating]]]:
    WAV_END_ADD = 1
    SAMPLE_LEN = 4

    wav_bytes = wav_helper.normalize_audio(wav_bytes)
    
    wav_len = wav_helper.wav_length_from_bytes(wav_bytes)

    intervals = birdnet_preprocess_interface.get_yellowhammer_intervals(wav_bytes)

    samples:List[Tuple[Tuple[float, float], np.ndarray[np.floating]]] = []

    logger.debug(f"trimming intervals: {intervals}")
    for start_sec, end_sec in intervals:
        end = wav_len if(end_sec + 1 > wav_len) else end_sec + 1
        samples.append(
            # we're trimming to end + treshold to hopefuly make sure dialect portion is whole
            (
                (start_sec, end),
                wav_helper.trim_audio_to_np_float(wav_bytes, start_sec, end, SAMPLE_LEN)
            )
        )
    logger.debug("trimming done")

    return samples


def preprocess_file(fp :str) -> List[Tuple[Tuple[float, float], np.ndarray[np.floating]]]:
    with open(fp, 'rb') as f:
        return preprocess(f.read())


def make_spects(samples) -> List[np.ndarray]:
    '''
    return list of spectrograms - 2d np ndarray - shape: (128, 376)
    all input samples should be PCM < -1; 1 > 48kHz mono np.ndarrays
    '''
    SHAPE = (128, 376)
    spects = []
    for s in samples:
        spects.append(
            spectrogram.mel_spectrogram(s)
        )
        if(spects[-1].shape != SHAPE): raise Exception("invalid spectrogram shape")

    return spects


# if __name__ == '__main__':
#     samples = preprocess('.tstdata/F003716.wav')
#     spects = make_spects(samples)
#     for s in spects:
        
#         print(s.shape)
#         spectrogram.plot_mel_spect(
#             s
#         )