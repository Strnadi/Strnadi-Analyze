import os
import sys
import shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import math
import glob
import keras
import librosa
import seaborn
import sklearn
import imblearn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import cache
from pathlib import Path

keras.config.set_dtype_policy("float32")

LOCAL_WORKSPACE = '/content'
WORKSPACE = '/content/drive/MyDrive/strnadi-data'

DATA_EXTENSIONS = [".npy"]
REMOTE_DATASET = os.path.join(WORKSPACE, 'strnadi-encoded.zip')
DATASET = os.path.join(LOCAL_WORKSPACE, 'dataset.zip')
DATASET_DIR = os.path.join(LOCAL_WORKSPACE, 'dataset')
SAMPLE_RATE, SAMPLE_SECONDS = 48000, 4
BATCH_SIZE = 32

shutil.copy(REMOTE_DATASET, DATASET)
!unzip -q -o $DATASET -d $DATASET_DIR

!rm $DATASET_DIR/.DS_Store

!find $DATASET_DIR -type f -name '*.npy' | wc -l

def data_generator(files, shuffle):
    """Generator that yields audio chunks and labels on demand"""
    indices = list(range(len(files)))

    if shuffle:
        np.random.shuffle(indices)

    for idx in indices:
        file_path: str = files[idx]
        data = np.load(file_path)

        if data is not None:
            data = data[0]
            data = np.expand_dims(data, axis=-1)
            yield data, data


def load_data(directory, validation_split=0.3, batch_size=BATCH_SIZE, shuffle=True):
    """
    Create a TensorFlow dataset from audio files in directory
    """
    audio_files = list(glob.glob(os.path.join(directory, f"*{DATA_EXTENSIONS[0]}")))
    print(f"Found {len(audio_files)} audio files")

    split_time = int(time.time())
    print(f"Dataset seed: {split_time}, {split_time+1}")

    # Split into train and validation sets
    train_files, val_files = sklearn.model_selection.train_test_split(
        audio_files, test_size=validation_split, random_state=split_time
    )

    val_files, test_files = sklearn.model_selection.train_test_split(
        val_files, test_size=0.33, random_state=split_time+1
    )

    print(f"Training on {len(train_files)} files, validating on {len(val_files)} files, testing on {len(test_files)} files")

    # Define output signature for the generator
    # This must match the expected input shape of the model (128, 376, 1)
    output_signature = (
        tf.TensorSpec(shape=(128, 376, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(128, 376, 1), dtype=tf.float32),
    )

    # Create TensorFlow datasets using generators
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_files, shuffle=shuffle),
        output_signature=output_signature
    )

    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(val_files, shuffle=False),
        output_signature=output_signature
    )

    # Apply batching and prefetching and caching
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset   = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_steps    = math.floor(len(train_files) / batch_size)
    val_steps      = math.floor(len(val_files) / batch_size)
    return train_dataset, val_dataset, train_steps, val_steps

train_dataset, val_dataset, train_steps, val_steps = load_data(DATASET_DIR)

class SpatialAttention(keras.layers.Layer):
    """
    CBAM-style Spatial Attention Module.
    It learns 'where' to look in the image (time/frequency) regardless of channels.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 7x7 convolution is standard for CBAM spatial attention to capture broader context
        self.conv = keras.layers.Conv2D(filters=1, kernel_size=kernel_size,
                                  padding='same', use_bias=False, activation='sigmoid')

    def call(self, inputs):
        # 1. Channel-wise Average Pooling
        avg_out = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # 2. Channel-wise Max Pooling
        max_out = tf.reduce_max(inputs, axis=-1, keepdims=True)
        # 3. Concatenate along channel axis
        x = keras.layers.Concatenate(axis=-1)([avg_out, max_out])
        # 4. Convolution -> Sigmoid Mask
        mask = self.conv(x)
        # 5. Apply mask to original inputs
        return inputs * mask

def build_yellowhammer_autoencoder(input_shape=(128, 376, 1), latent_dim=64):
    """
    Constructs the Autoencoder with Asymmetric Pooling and Spatial Attention.
    """
    # ==========================
    # ENCODER
    # ==========================
    inputs = keras.Input(shape=input_shape)

    # Block 1: Standard reduction
    x = keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    # Shape: (64, 188, 32)

    # Block 2: Asymmetric Pooling (Pool Time only, Preserve Freq)
    # Critical for distinguishing Bh (High) vs Bl (Low) dialects
    x = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D((1, 2))(x)
    # Shape: (64, 94, 64) -> Frequency height remains 64

    # Block 3: Deep features
    x = keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)

    # === SPATIAL ATTENTION BLOCK ===
    # The model learns to multiply the "trill" (left side) by ~0
    # and the "dialect" (right side) by ~1 here.
    x = SpatialAttention(kernel_size=7)(x)

    # Final reduction before latent
    x = keras.layers.MaxPooling2D((2, 2))(x)
    # Shape: (32, 47, 128)

    # Flatten and Latent Vector
    # Save volume shape for decoder reshaping
    volume_shape = x.shape[1:]
    x = keras.layers.Flatten()(x)
    latent = keras.layers.Dense(latent_dim, name='latent_vector')(x)

    # ==========================
    # DECODER
    # ==========================
    # Project back to volume
    x = keras.layers.Dense(volume_shape[0] * volume_shape[1] * volume_shape[2])(latent)
    x = keras.layers.Reshape(volume_shape)(x)

    # Mirror Block 3
    x = keras.layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Mirror Block 2 (Asymmetric Upsample)
    x = keras.layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.UpSampling2D((1, 2))(x) # Upsample Time only

    # Mirror Block 1
    x = keras.layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    # Final reconstruction
    outputs = keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(x)

    # ==========================
    # MODELS
    # ==========================
    autoencoder = keras.Model(inputs, outputs, name="autoencoder")
    encoder = keras.Model(inputs, latent, name="encoder")

    return autoencoder, encoder

def mbconv_block(inputs, expansion_factor, filters, stride, kernel_size=3):
    """
    Mobile Inverted Residual Bottleneck Block (MBConv).
    Includes: Expansion -> Depthwise Conv -> Pointwise Conv (Projection) + Skip Connection
    """
    input_channels = inputs.shape[-1]
    expanded_channels = input_channels * expansion_factor

    # 1. Expansion Phase (1x1 Conv)
    if expansion_factor != 1:
        x = keras.layers.Conv2D(expanded_channels, kernel_size=1, padding='same', use_bias=False)(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(6.0)(x) # ReLU6 is standard for MobileNet
    else:
        x = inputs

    # 2. Depthwise Convolution (Spatial features)
    x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.0)(x)

    # 3. Pointwise Convolution (Linear Projection - No Activation)
    x = keras.layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)

    # 4. Skip Connection (Residual)
    # Only if input/output shapes match (stride=1 and channels equal)
    if stride == 1 and input_channels == filters:
        x = keras.layers.Add()([inputs, x])

    return x

def build_mobile_autoencoder(input_shape=(128, 376, 1), latent_dim=64):
    inputs = keras.Input(shape=input_shape)

    # ==========================
    # ENCODER (MobileNetV2 Style)
    # ==========================
    # Initial stem
    x = keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs) # (64, 188)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.0)(x)

    # MBConv Blocks
    # Note: We use strides inside the blocks to downsample instead of MaxPool

    # Block 1: Capture low-level edges
    x = mbconv_block(x, expansion_factor=1, filters=16, stride=1)

    # Block 2: Downsample Asymmetrically (Freq=1, Time=2) ?
    # Standard MBConv stride is symmetric (2,2). For your specific "Frequency Shape" need,
    # we might strictly separate stride or use standard (2,2) if the spectrogram is large enough.
    # Let's stick to standard (2,2) for efficiency, or (1,2) if you define custom stride tuple.

    # Standard Mobile stride (2,2)
    x = mbconv_block(x, expansion_factor=6, filters=24, stride=2) # (32, 94)
    x = mbconv_block(x, expansion_factor=6, filters=24, stride=1) # Refine

    # Block 3: Deeper Features
    x = mbconv_block(x, expansion_factor=6, filters=32, stride=2) # (16, 47)
    x = mbconv_block(x, expansion_factor=6, filters=32, stride=1)

    # Block 4: The "Semantic" Level
    x = mbconv_block(x, expansion_factor=6, filters=64, stride=1)

    # === SPATIAL ATTENTION (Keep this!) ===
    # Even with MBConv, this is crucial for the "Trill vs Dialect" problem
    # (Reuse the SpatialAttention class from previous response)
    x = SpatialAttention()(x)

    # Latent Vector
    x = keras.layers.GlobalAveragePooling2D()(x) # More efficient than Flatten+Dense
    latent = keras.layers.Dense(latent_dim, name='latent_vector')(x)

    # ==========================
    # DECODER
    # ==========================
    # For the decoder, standard Conv2DTranspose is usually fine and simpler to implement
    # than "Inverted Residual Transpose".

    # Reshape back to the volume output of the encoder (approximate)
    x = keras.layers.Dense(16 * 47 * 64)(latent)
    x = keras.layers.Reshape((16, 47, 64))(x)

    x = keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x) # (32, 94)
    x = keras.layers.Conv2DTranspose(24, 3, strides=2, padding='same', activation='relu')(x) # (64, 188)
    x = keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x) # (128, 376)

    outputs = keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    return keras.Model(inputs, outputs), keras.Model(inputs, latent)

# ---------------------------------------------------------
# Usage Example
# ---------------------------------------------------------

# 1. Build Model
autoencoder, encoder = build_yellowhammer_autoencoder()

# 2. Compile
# Use MSE because we want pixel-perfect reconstruction of the frequency lines.
autoencoder.compile(optimizer='adam', loss='mse')

# 3. Summary to verify shapes
autoencoder.summary()

# ---------------------------------------------------------
# How to run Unsupervised Clustering
# ---------------------------------------------------------
# Assuming 'X_train' is your numpy array of spectrograms scaled 0-1
# Shape: (num_samples, 128, 376, 1)

# Step A: Train the Autoencoder to reconstruct the images
# autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1)

# Step B: Extract embeddings (Latent Vectors)
# latent_vectors = encoder.predict(X_train)

# Step C: Cluster (using HDBSCAN or GMM)
# import hdbscan
# clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
# labels = clusterer.fit_predict(latent_vectors)


mobile_autoencoder, mobile_encoder = build_mobile_autoencoder()
# mobile_autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2), loss='mse')
mobile_autoencoder.compile(optimizer='adam', loss='mse')
mobile_autoencoder.summary()

EPOCHS = 100

current_time = int(time.time())

# Make checkpoint dir
checkpoint_dir = os.path.join(WORKSPACE, 'checkpoints', str(current_time))
tensorboard_dir = os.path.join(WORKSPACE, 'tensorboard', str(current_time))
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)

backup_dir = os.path.join(WORKSPACE, 'training-backups', str(current_time))

history = autoencoder.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, '{epoch}-{val_loss:.4f}.keras'),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            save_freq="epoch"
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.BackupAndRestore(backup_dir, double_checkpoint=True),
        keras.callbacks.TensorBoard(
            log_dir=tensorboard_dir
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
)
