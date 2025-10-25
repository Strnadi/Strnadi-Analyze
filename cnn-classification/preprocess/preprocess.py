from . import birdnet_preprocess_interface
from . import wav_helper
from . import spectrogram

import numpy as np


def preprocess(fp:str):
    WAV_END_ADD = 1
    SAMPLE_LEN = 4

    wav_bytes = None
    with open(fp, 'rb') as f:
        wav_bytes = wav_helper.normalize_audio(f.read())
    
    wav_len = wav_helper.wav_length_from_bytes(wav_bytes)

    intervals = birdnet_preprocess_interface.get_yellowhammer_intervals(wav_bytes)
    print(intervals)

    samples:list[np.NDArray[np.floating[np.Any]]] = []

    for start_sec, end_sec in intervals:
        end = wav_len if(end_sec + 1 > wav_len) else end_sec + 1
        samples.append(
            # we're trimming to end + treshold to hopefuly make sure dialect portion is whole
            wav_helper.trim_audio_to_np_float(wav_bytes, start_sec, end, SAMPLE_LEN)
        )

    return samples


def make_spects(samples) -> list[np.ndarray]:
    '''
    return list of spectrograms - 2d np NDArray - shape: (128, 376)
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


if __name__ == '__main__':
    samples = preprocess('.tstdata/F003716.wav')
    spects = make_spects(samples)
    for s in spects:
        
        print(s.shape)
        spectrogram.plot_mel_spect(
            s
        )