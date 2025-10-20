import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
import matplotlib.pyplot as plt
from functools import partial


BATCH_SIZE = 8
SAMPLE_RATE = 16000
EPOCHS = 50
SAMPLE_SECONDS = 4
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
        if count > 0:  # Avoid division by zero
            # Formula: total_samples / (num_classes * samples_in_class)
            class_weights[i] = total_samples / (len(class_names) * count)
    
    print("Class weights:", class_weights)
    return class_weights


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
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset, class_names, class_weights


ds_train, ds_validate, class_names, class_weights = load_data(directory)
num_classes = len(class_names)


def build_model(sample_seconds=SAMPLE_SECONDS):
    audio_in = keras.Input(shape=(SAMPLE_RATE * sample_seconds,), name="audio_in")

    spectrogram_layer = keras.layers.MelSpectrogram(
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
    )
    spectrogram = spectrogram_layer(audio_in)

    # spectrogram = StaticNoiseReducer(percentile=0.1, floor_margin_db=5.0, name="spectral_noise_gate")(spectrogram)

    # Add a channel dimension
    reshaped_spectrogram = keras.layers.Reshape((*spectrogram.shape[1:], 1))(spectrogram)

    # Convert single-channel spectrogram to 3 channels for ResNet
    three_channel_spectrogram = keras.layers.Concatenate(axis=-1)(
        [reshaped_spectrogram, reshaped_spectrogram, reshaped_spectrogram]
    )

    # 3) ResNet50 for feature extraction
    # resnet_base = keras.applications.ResNet50V2(
    #     include_top=False,
    #     weights=None,
    #     pooling="avg"
    # )
    # feat_resnet = resnet_base(three_channel_spectrogram)

    effinet_base = keras.applications.EfficientNetV2B0(
        include_top=False,
        weights=None,
        pooling="avg"
    )
    feat_efficient = effinet_base(three_channel_spectrogram)

    # # Encoder
    # encoder = keras.layers.Dense(128, activation="relu", name="encoder_1")(feat_resnet)
    # latent = keras.layers.Dense(64, activation="relu", name="latent")(encoder)

    # # Decoder
    # decoder = keras.layers.Dense(128, activation="relu", name="decoder_1")(latent)
    # reconstruction = keras.layers.Dense(feat_resnet.shape[-1], activation="linear", name="reconstruction")(decoder)

    # dialect_classify_hidden = keras.layers.Dense(64, activation="relu", name="bird_classify_hidden")(feat_resnet)
    dialect_classify = keras.layers.Dense(num_classes, activation="softmax", name="bird_classify")(feat_efficient)

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
    metrics=["accuracy", keras.metrics.AUC(name="auc")]
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
    # validation_data=ds_validate,
    class_weight=class_weights,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        # keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        # keras.callbacks.TensorBoard(log_dir="./logs")
    ]
)

# Save the model
model.save("bird_dialect_model.keras")

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
