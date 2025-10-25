import os
import sys
import keras
import scipy, scipy.io, scipy.signal
import numpy as np
import tensorflow as tf
import librosa
import glob
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

BATCH_SIZE = 16
SAMPLE_RATE = 16000
EPOCHS = 50
SAMPLE_SECONDS = 4
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]

# Directory with audio files
directory = "/home/hroudis/strnad-data"

# Custom audio loading and processing functions
def load_and_normalize_audio(file_path, target_sr=SAMPLE_RATE):
    """
    Load audio file in various formats and normalize it
    """
    try:
        # Use librosa to handle various audio formats
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        # Normalize audio to [-1, 1] range
        audio = librosa.util.normalize(audio)
        
        return audio, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def process_audio_file(file_path, target_seconds=SAMPLE_SECONDS, target_sr=SAMPLE_RATE):
    """
    Process audio file and split into chunks of target_seconds length
    """
    audio, sr = load_and_normalize_audio(file_path, target_sr)
    if audio is None:
        return []
    
    target_length = int(target_seconds * target_sr)
    chunks = []
    
    # If audio is shorter than target length, pad with zeros
    if len(audio) < target_length:
        padded_audio = np.zeros(target_length)
        padded_audio[:len(audio)] = audio
        chunks.append(padded_audio)
    else:
        # Split audio into chunks of target_length
        num_chunks = int(np.ceil(len(audio) / target_length))
        for i in range(num_chunks):
            start = i * target_length
            end = min(start + target_length, len(audio))
            
            chunk = np.zeros(target_length)
            chunk[:(end-start)] = audio[start:end]
            chunks.append(chunk)
    
    return chunks

def audio_generator(files, labels, class_names, shuffle):
    """Generator that yields audio chunks and labels on demand"""
    num_classes = len(class_names)
    indices = list(range(len(files)))
    
    if shuffle:
        np.random.shuffle(indices)
    
    for idx in indices:
        file_path = files[idx]
        label = labels[idx]
        
        chunks = process_audio_file(file_path)
        if not chunks:
            continue
            
        for chunk in chunks:
            # One-hot encode the label
            one_hot = np.zeros(num_classes)
            one_hot[label] = 1
            
            # Bird presence is always 1 for now (assuming all files contain birds)
            bird_label = 1.0
            
            yield chunk, {"bird_detection": bird_label, "dialect_classifier": one_hot}

def create_audio_dataset(directory, validation_split=0.3, batch_size=BATCH_SIZE, shuffle=True):
    """
    Create a TensorFlow dataset from audio files in directory
    """
    audio_files = []
    class_names = []
    labels = []
    
    subdirs = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    class_names = sorted(subdirs)
    class_indices = {name: i for i, name in enumerate(class_names)}
    
    # For each class directory, find all audio files
    for subdir in subdirs:
        class_dir = os.path.join(directory, subdir)
        class_idx = class_indices[subdir]
        
        for ext in AUDIO_EXTENSIONS:
            pattern = os.path.join(class_dir, f"*{ext}")
            audio_paths = glob.glob(pattern)
            
            for path in audio_paths:
                audio_files.append(path)
                labels.append(class_idx)
    
    # Split into train and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=validation_split, stratify=labels, random_state=42
    )
    
    print(f"Found {len(audio_files)} audio files in {len(class_names)} classes")
    print(f"Training on {len(train_files)} files, validating on {len(val_files)} files")

    # Define output signature for the generator
    output_signature = (
        tf.TensorSpec(shape=(SAMPLE_RATE * SAMPLE_SECONDS,), dtype=tf.float32),
        {
            "bird_detection": tf.TensorSpec(shape=(), dtype=tf.float32),
            # "dialect_classifier": tf.TensorSpec(shape=(len(class_names),), dtype=tf.float32)
        }
    )
    
    # Create TensorFlow datasets using generators
    train_dataset = tf.data.Dataset.from_generator(
        lambda: audio_generator(train_files, train_labels, class_names, shuffle=shuffle),
        output_signature=output_signature
    )
    
    val_dataset = tf.data.Dataset.from_generator(
        lambda: audio_generator(val_files, val_labels, class_names, shuffle=False),
        output_signature=output_signature
    )
    
    # Apply batching and prefetching
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset, class_names

# Create custom datasets instead of using keras.utils.audio_dataset_from_directory
ds_train, ds_validate, class_names = create_audio_dataset(directory)

NUM_CLASSES = len(class_names)


def build_bird_dialect_model(sample_seconds=SAMPLE_SECONDS):
    audio_in = keras.Input(shape=(SAMPLE_RATE * sample_seconds,), name="audio_input")

    spectrogram_layer = keras.layers.MelSpectrogram(
        max_freq=8000,
        min_freq=3500,
        fft_length=2048,
        sequence_stride=512,
        window="hann",
        sampling_rate=SAMPLE_RATE,
        sequence_length=sample_seconds * SAMPLE_RATE,
        num_mel_bins=128,
        power_to_db=True,
        top_db=80.0,
        name="mel_spectrogram"
    )
    spectrogram = spectrogram_layer(audio_in)

    # 3) ResNet50 for feature extraction
    resnet_base = keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        pooling="avg"
    )
    feat_resnet = resnet_base(spectrogram)

    bird_detect = keras.layers.Dense(64, activation="relu", name="bird_detect_hidden")(feat_resnet)
    bird_presence = keras.layers.Dense(1, activation="sigmoid", name="bird_detection")(bird_detect)
    
    return keras.Model(
        inputs=audio_in,
        outputs={
            "bird_detection": bird_presence,
            # "dialect_classifier": dialect_output,
            # "reconstruction": reconstruction,
            # "latent_vector": latent_output
        },
        name="bird_dialect_model"
    )

model = build_bird_dialect_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        "bird_detection": "binary_crossentropy",
        "dialect_classifier": "categorical_crossentropy",
        "reconstruction": "mse"
    },
    loss_weights={
        "bird_detection": 0.3,
        "dialect_classifier": 1.0,
        "reconstruction": 0.1
    },
    metrics={
        "bird_detection": ["accuracy", keras.metrics.AUC(name="auc")],
        "dialect_classifier": ["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    }
)

model.summary()

# Update the training code to work with our custom dataset
# history = model.fit(
#     ds_train,
#     validation_data=ds_validate,
#     epochs=EPOCHS,
#     callbacks=[
#         keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
#         keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
#         keras.callbacks.TensorBoard(log_dir="./logs")
#     ]
# )

# Save the model
# model.save("bird_dialect_model.keras")

# Also save some metadata about the classes for future inference
# with open("model_metadata.json", "w") as f:
#     json.dump({
#         "class_names": class_names,
#         "sample_rate": SAMPLE_RATE,
#         "sample_seconds": SAMPLE_SECONDS
#     }, f)
