from ai_edge_litert.interpreter import Interpreter
import os
import librosa
import numpy as np
import sys
import time
from tqdm import tqdm
sys.path.insert(0, os.path.join("/workspace", "Strnadi-Analyze"))

from evaluation.file_names import collect_dataset

MODEL_PATH = "perch_v2_based.tflite"
WORKSPACE = '/workspace'

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]
DATASET_DIR = os.path.join(WORKSPACE, 'dataset')
SAMPLE_RATE, SAMPLE_SECONDS = 48000, 4
BATCH_SIZE = 32

def load_and_normalize_audio(file_path, target_sr=32000, target_duration=5):
    """
    Load audio file in various formats and normalize it.
    Returns a 1D numpy array or None on error.
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
        if audio.ndim > 1:
            # if multi-channel, average channels unless second channel is all zeros
            if audio.shape[0] > 1 and np.any(audio[1]):
                audio = np.mean(audio, axis=0)
            else:
                audio = audio[0]

        if len(audio) < target_duration * target_sr:
            audio = np.pad(audio, (0, int(target_duration * target_sr - len(audio))), mode='constant')
        elif len(audio) > target_duration * target_sr:
            audio = audio[:int(target_duration * target_sr)]

        audio = librosa.util.normalize(audio)
        return audio

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def audio_generator(files, labels, class_names, shuffle):
    """Generator that yields audio chunks and labels on demand"""
    num_classes = len(class_names)
    indices = list(range(len(files)))

    if shuffle:
        np.random.shuffle(indices)

    for idx in indices:
        file_path: str = files[idx]
        label = labels[idx]  # This is the integer label

        # One-hot encode the label for loss calculation
        one_hot = np.zeros(num_classes)
        one_hot[label] = 1
        audio = load_and_normalize_audio(file_path, target_sr=32000, target_duration=5)

        if audio is not None:
            # Yield ((audio_input, integer_label_input), one_hot_label_for_loss)
            # yield ((audio, label), one_hot)
            yield audio, one_hot

interpreter = Interpreter(model_path=MODEL_PATH, num_threads=16)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# print(input_details)
# print(output_details)

files, labels = collect_dataset(dataset_dir=DATASET_DIR, validation_split=0.3, train=True, val=True, test=True, seed=42)
# class_names = [f.name for f in os.scandir(DATASET_DIR) if f.is_dir()]

outputs = []

for audio, label in tqdm(audio_generator(files, labels, ["BC", "BE", "BhBl", "BlBh", "None", "Unfinished", "XB"], shuffle=True)):
    interpreter.set_tensor(input_details[0]['index'], np.array([audio]))
    interpreter.invoke()
    interpreter.get_tensor(output_details[0]['index'])
    # outputs.append(interpreter.get_tensor(output_details[0]['index']))

# outputs = np.array(outputs)
