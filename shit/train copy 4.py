import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math
import keras
import librosa
import numpy as np
import tensorflow as tf
import scipy, scipy.io, scipy.signal
import glob
import json
import ffmpeg
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
from sklearn.model_selection import train_test_split


BATCH_SIZE = 8
SAMPLE_RATE = 32000
EPOCHS = 50
SAMPLE_SECONDS = 3.5
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]
directory = "/home/hroudis/strnad-data"


def load_and_normalize_audio(file_path, target_sr=SAMPLE_RATE):
    """
    Load audio file in various formats and normalize it
    """
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
        if audio.ndim > 1:
            if audio.shape[0] > 1 and np.any(audio[1]):
                audio = np.mean(audio, axis=0)
            else:
                audio = audio[0]

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

    # Apply high-pass filter to remove low-frequency noise
    # cutoff_hz = 11000.0
    # sos = scipy.signal.butter(4, cutoff_hz, btype='highpass', fs=target_sr, output='sos')
    # filtered_audio = scipy.signal.sosfilt(sos, audio)

    # Calculate RMS on the filtered audio
    rms = np.sqrt(np.mean(audio**2))
    # volume_factor = 10 ** ((rms) / 50) * 0.02
    # volume_factor = max(0.005, min(volume_factor, 0.05))

    volume_factor = rms / (10 ** (30.0 / 20))

    target_length = int(target_seconds * target_sr)
    chunks = []

    # If audio is shorter than target length, pad with noise
    if len(audio) < target_length:
        padded_audio = np.zeros(target_length)
        padded_audio[:len(audio)] = audio

        # Generate random noise for padding
        noise = np.random.normal(0, volume_factor, target_length - len(audio))

        # Add the noise to the padding area
        padded_audio[len(audio):] = noise
        chunks.append(padded_audio)

    else:
        # Split audio into chunks of target_length
        num_chunks = int(np.ceil(len(audio) / target_length))
        for i in range(num_chunks):
            start = i * target_length
            end = min(start + target_length, len(audio))

            # Get the audio slice for this chunk
            chunk = audio[start:end]

            # If this chunk is not completely filled, pad with noise
            if len(chunk) < target_length:
                noise = np.random.normal(0, volume_factor, target_length - len(chunk))
                chunk = np.concatenate([chunk, noise])

            chunks.append(chunk)

    return chunks


def compute_class_weights(labels, class_names):
    """
    Compute class weights inversely proportional to class frequencies
    """
    # Count samples per class
    class_counts = np.bincount(labels, minlength=len(class_names))

    # Calculate weights inversely proportional to counts
    total_samples = len(labels)
    class_weights = {}

    for i, count in enumerate(class_counts):
        if count > 0:
            # Formula: total_samples / (num_classes * samples_in_class)
            class_weights[i] = total_samples / (len(class_names) * count)

    print("Class weights:", class_weights)
    return class_weights


def audio_generator(files, labels, class_names, shuffle):
    """Generator that yields audio chunks and labels on demand"""
    num_classes = len(class_names)
    indices = list(range(len(files)))

    if shuffle:
        np.random.shuffle(indices)

    for idx in indices:
        file_path: str = files[idx]
        label = labels[idx]

        chunks = process_audio_file(file_path)
        if not chunks:
            continue

        for chunk in chunks:
            # One-hot encode the label
            one_hot = np.zeros(num_classes)
            one_hot[label] = 1
            
            # Only yield the outputs that match the defined output signature
            yield chunk, one_hot


def load_data(directory, validation_split=0.3, batch_size=BATCH_SIZE, shuffle=True):
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

    # Calculate class weights before splitting
    class_weights = compute_class_weights(labels, class_names)

    # Split into train and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        audio_files, labels, test_size=validation_split, stratify=labels, random_state=42
    )

    print(f"Found {len(audio_files)} audio files in {len(class_names)} classes")
    print(f"Training on {len(train_files)} files, validating on {len(val_files)} files")

    # Define output signature for the generator
    output_signature = (
        tf.TensorSpec(shape=(SAMPLE_RATE * SAMPLE_SECONDS,), dtype=tf.float32),
        tf.TensorSpec(shape=(len(class_names),), dtype=tf.float32)
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
    val_dataset   = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_steps    = math.ceil(len(train_files) / batch_size)
    val_steps      = math.ceil(len(val_files)   / batch_size)
    return train_dataset, val_dataset, class_names, class_weights, train_steps, val_steps


ds_train, ds_validate, class_names, class_weights, train_steps, val_steps = load_data(directory)
num_classes = len(class_names)


def build_model(sample_seconds=SAMPLE_SECONDS, denoise_model: keras.Model = None):
    audio_in = keras.Input(shape=(SAMPLE_RATE * sample_seconds,), name="audio_in")

    spectrogram = keras.layers.MelSpectrogram(
        min_freq=4000,
        max_freq=9000,
        num_mel_bins=512,
        sequence_length=1024,
        sequence_stride=128,
        fft_length=4 * SAMPLE_RATE,
        sampling_rate=SAMPLE_RATE,
        power_to_db=True,
        window="hann",
        name="mel_spectrogram"
    )(audio_in)

    # spectrogram = denoise_model(spectrogram)

    # spectrogram = StaticNoiseReducer(percentile=0.1, floor_margin_db=5.0, name="spectral_noise_gate")(spectrogram)

    spectrogram = keras.layers.Reshape((*spectrogram.shape[1:], 1))(spectrogram)
    spectrogram = keras.layers.Concatenate(axis=-1)(
        [spectrogram, spectrogram, spectrogram]
    )

    cnn = keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )(spectrogram)

    # projection = keras.layers.Dense(128, activation="relu")(cnn)
    # projection = keras.layers.Dense(64, activation="relu")(projection)
    # projection = keras.layers.Dense(32, activation="relu")(projection)

    # # Final dimensions for autoencoder: 2D plottable space (x, y)
    # projection = keras.layers.Dense(2, name="projection")(projection)

    # # 2. Add L2 normalization for contrastive learning
    # normalized_projection = keras.layers.Lambda(
    #     lambda x: tf.math.l2_normalize(x, axis=1),
    #     name="normalized_projection"
    # )(projection)

    # # 3. Decoder for reconstruction objective
    # decoder = keras.layers.Dense(32, activation="relu")(projection)
    # decoder = keras.layers.Dense(64, activation="relu")(decoder)
    # decoder = keras.layers.Dense(128, activation="relu")(decoder)
    # reconstruction = keras.layers.Dense(cnn.shape[-1])(decoder)

    # dialect_classify_hidden = keras.layers.Dense(64, activation="relu")(cnn)
    dialect_classify = keras.layers.Dense(num_classes, activation="softmax")(cnn)

    return keras.Model(
        inputs=audio_in,
        outputs=dialect_classify,
        # outputs={
        #     "dialect": dialect_classify,
        #     # "vector": latent,
        # }
    )


model = build_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", "auc"]
    # loss={
    #     "dialect": keras.losses.CategoricalCrossentropy(from_logits=True),
    #     # "dialect_classifier": "categorical_crossentropy",
    #     # "reconstruction": "mse"
    # },
    # loss_weights={
    #     "dialect": 1.0,
    #     # "dialect_classifier": 1.0,
    #     # "reconstruction": 0.1
    # },
    # metrics={
    #     "bird_detection": ["accuracy", keras.metrics.AUC(name="auc")],
    #     # "dialect_classifier": ["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    # }
)

model.summary()

keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_layer_activations=True,
    show_trainable=True
)


# Update the training code to work with our custom dataset
model.fit(
    ds_train,
    validation_data=ds_validate,
    class_weight=class_weights,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        # keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        # keras.callbacks.TensorBoard(log_dir="./logs")
    ]
)

# Save the model
model.save("model.keras")

# Also save some metadata about the classes for future inference
with open("model_metadata.json", "w") as f:
    json.dump(
        {
            "class_names": class_names,
            "sample_rate": SAMPLE_RATE,
            "sample_seconds": SAMPLE_SECONDS
        },
        f
    )
