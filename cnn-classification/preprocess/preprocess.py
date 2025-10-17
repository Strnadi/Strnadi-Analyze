import birdnet_preprocess_interface
import wav_helper
import numpy as np

import spectrogram

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


# if __name__ == '__main__':
#     samples = preprocess('.tstdata/F000252.wav')
#     for s in samples:
#         spectrogram.plot_mel_spect(
#             spectrogram.mel_spectrogram(s)
#         )