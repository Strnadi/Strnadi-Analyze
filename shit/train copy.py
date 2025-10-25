import os
import sys
import keras
import scipy, scipy.io, scipy.signal
import numpy as np
import tensorflow as tf
import librosa
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# TODO: Classification class weights based on file counts

BATCH_SIZE = 16
SAMPLE_RATE = 16000
EPOCHS = 50
SAMPLE_SECONDS = 4
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]
directory = "/home/hroudis/strnad-data"

# Custom audio loading and processing functions
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
            
            # Bird presence is a float32 (1.0 or 0.0) rather than a boolean
            bird_label = 1.0 if file_path.find("NoDialect") == -1 else 0.0
            
            # Only yield the outputs that match the defined output signature
            yield chunk, {"bird_detection": bird_label}
            # yield chunk, {"bird_detection": bird_label, "dialect_classifier": one_hot}

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
        min_freq=4400,
        max_freq=9000,
        num_mel_bins=512,
        sequence_length=1024,
        sequence_stride=128,
        fft_length=4 * SAMPLE_RATE,
        sampling_rate=SAMPLE_RATE,
        power_to_db=True,
        min_power=0.5,
        window="hann",
        name="mel_spectrogram"
    )
    spectrogram = spectrogram_layer(audio_in)

    # Add a channel dimension
    reshaped_spectrogram = keras.layers.Reshape((*spectrogram.shape[1:], 1))(spectrogram)

    # Convert single-channel spectrogram to 3 channels for ResNet
    three_channel_spectrogram = keras.layers.Concatenate(axis=-1)(
        [reshaped_spectrogram, reshaped_spectrogram, reshaped_spectrogram]
    )

    # 3) ResNet50 for feature extraction
    resnet_base = keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        pooling="avg"
    )
    feat_resnet = resnet_base(three_channel_spectrogram)

    # Encoder
    encoder = keras.layers.Dense(128, activation="relu", name="encoder_1")(feat_resnet)
    latent = keras.layers.Dense(64, activation="relu", name="latent")(encoder)

    # Decoder
    decoder = keras.layers.Dense(128, activation="relu", name="decoder_1")(latent)
    reconstruction = keras.layers.Dense(feat_resnet.shape[-1], activation="linear", name="reconstruction")(decoder)


    # bird_detect = keras.layers.Dense(64, activation="relu", name="bird_detect_hidden")(feat_resnet)
    # bird_presence = keras.layers.Dense(1, activation="sigmoid", name="bird_detection")(bird_detect)

    # # 4) Vision Transformer branch (processes spectrogram if bird is present)
    # # Simplified ViT-like processing with attention
    # # Reshape spectrogram for patch embedding
    # spec_reshaped = keras.layers.Reshape((-1, 128), name="vit_reshape")(spectrogram)

    # # Patch embedding
    # vit_embed = keras.layers.Dense(256, activation="relu", name="vit_embed")(spec_reshaped)

    # # Multi-head attention
    # vit_attention = keras.layers.MultiHeadAttention(
    #     num_heads=8, 
    #     key_dim=32,
    #     name="vit_attention"
    # )(vit_embed, vit_embed)

    # # Global average pooling
    # vit_pooled = keras.layers.GlobalAveragePooling1D(name="vit_pool")(vit_attention)
    # feat_vit = keras.layers.Dense(128, activation="relu", name="vit_features")(vit_pooled)

    # # Weight ViT features by bird detection confidence
    # weighted_vit = keras.layers.Multiply(name="weighted_vit")([feat_vit, bird_presence])

    # # 5) Fuse features
    # fused = keras.layers.Concatenate(name="fusion")([feat_resnet, weighted_vit])

    # # 6) Autoencoder for dimension reduction
    # # Encoder
    # encoder = keras.layers.Dense(128, activation="relu", name="encoder_1")(fused)
    # latent = keras.layers.Dense(64, activation="relu", name="latent")(encoder)

    # # Decoder
    # decoder = keras.layers.Dense(128, activation="relu", name="decoder_1")(latent)
    # reconstruction = keras.layers.Dense(fused.shape[-1], activation="linear", name="reconstruction")(decoder)

    # # 7) Final classifier on latent space
    # classifier_hidden = keras.layers.Dense(128, activation="relu", name="classifier_hidden")(latent)
    # classifier_dropout = keras.layers.Dropout(0.3, name="classifier_dropout")(classifier_hidden)
    # dialect_output = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="dialect_classifier")(classifier_dropout)

    return keras.Model(
        inputs=audio_in,
        outputs={
            "bird_detection": bird_presence,
            # "dialect_classifier": dialect_output,
            # "reconstruction": reconstruction,
            # "latent_vector": latent
        },
        name="bird_dialect_model"
    )


model = build_bird_dialect_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        "bird_detection": "binary_crossentropy",
        # "dialect_classifier": "categorical_crossentropy",
        # "reconstruction": "mse"
    },
    # loss_weights={
    #     "bird_detection": 1.0,
    #     # "dialect_classifier": 1.0,
    #     # "reconstruction": 0.1
    # },
    metrics={
        "bird_detection": ["accuracy", keras.metrics.AUC(name="auc")],
        # "dialect_classifier": ["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    }
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
# model.fit(
#     ds_train,
#     # train_test_split=0.2,
#     # ds_validate,
#     epochs=EPOCHS,
#     callbacks=[
#         keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
#         # keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
#         # keras.callbacks.TensorBoard(log_dir="./logs")
#     ]
# )

# Save the model
model.save("bird_dialect_model.keras")

# Also save some metadata about the classes for future inference
# with open("model_metadata.json", "w") as f:
#     json.dump({
#         "class_names": class_names,
#         "sample_rate": SAMPLE_RATE,
#         "sample_seconds": SAMPLE_SECONDS
#     }, f)
