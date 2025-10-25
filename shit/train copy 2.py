import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
from functools import partial

# TODO: Classification class weights based on file counts

BATCH_SIZE = 16
SAMPLE_RATE = 16000
EPOCHS = 50
SAMPLE_SECONDS = 4
AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".ogg", ".aiff"]
directory = "/home/hroudis/strnad-data"

# class StaticNoiseReducer(keras.layers.Layer):
#     def __init__(self, percentile=0.1, floor_margin_db=5.0, reduction_axis=1, epsilon=1e-4, **kwargs):
#         super().__init__(**kwargs)
#         self.percentile = percentile
#         self.floor_margin_db = floor_margin_db
#         self.reduction_axis = reduction_axis
#         self.epsilon = epsilon

#     def call(self, inputs):
#         sorted_vals = tf.sort(inputs, axis=self.reduction_axis)
#         axis_len = tf.shape(sorted_vals)[self.reduction_axis]
#         idx = tf.cast(tf.round(self.percentile * tf.cast(axis_len - 1, tf.float32)), tf.int32)
#         noise_floor = tf.gather(sorted_vals, idx, axis=self.reduction_axis)
#         noise_floor = tf.expand_dims(noise_floor, axis=self.reduction_axis)
#         threshold = noise_floor + self.floor_margin_db
#         return tf.maximum(inputs - threshold, self.epsilon)

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "percentile": self.percentile,
#                 "floor_margin_db": self.floor_margin_db,
#                 "reduction_axis": self.reduction_axis,
#                 "epsilon": self.epsilon,
#             }
#         )
#         return config

# def load_dataset(directory, batch_size=BATCH_SIZE, sample_rate=SAMPLE_RATE, sample_seconds=SAMPLE_SECONDS, validation_split=0.2):
#     ds_train, ds_validate = keras.utils.audio_dataset_from_directory(
#         directory,
#         batch_size=batch_size,
#         sampling_rate=sample_rate,
#         output_sequence_length=sample_rate * sample_seconds,
#         validation_split=validation_split,
#         subset="both",
#         verbose=True,
#         shuffle=False
#     )

#     def to_multi_targets(dataset, waveform, label):
#         dialect = tf.one_hot(label, len(dataset.class_names))
#         return waveform, {"dialect": dialect}

#     ds_train = ds_train.map(partial(to_multi_targets, ds_train))
#     ds_validate = ds_validate.map(partial(to_multi_targets, ds_validate))
#     return ds_train, ds_validate

ds_train, ds_validate = keras.utils.audio_dataset_from_directory(
    directory,
    batch_size=BATCH_SIZE,
    sampling_rate=SAMPLE_RATE,
    output_sequence_length=SAMPLE_RATE * SAMPLE_SECONDS,
    validation_split=0.2,
    subset="both",
    verbose=True,
    shuffle=False
)

num_classes = len(ds_train.class_names)

def to_one_hot(waveform, label):
    return waveform, tf.one_hot(label, depth=num_classes)

ds_train     = ds_train.map(to_one_hot)
ds_validate  = ds_validate.map(to_one_hot)

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
    resnet_base = keras.applications.ResNet50V2(
        include_top=False,
        weights=None,
        pooling="avg"
    )
    feat_resnet = resnet_base(three_channel_spectrogram)

    # # Encoder
    # encoder = keras.layers.Dense(128, activation="relu", name="encoder_1")(feat_resnet)
    # latent = keras.layers.Dense(64, activation="relu", name="latent")(encoder)

    # # Decoder
    # decoder = keras.layers.Dense(128, activation="relu", name="decoder_1")(latent)
    # reconstruction = keras.layers.Dense(feat_resnet.shape[-1], activation="linear", name="reconstruction")(decoder)

    dialect_classify_hidden = keras.layers.Dense(64, activation="relu", name="bird_classify_hidden")(feat_resnet)
    dialect_classify = keras.layers.Dense(num_classes, activation="softmax", name="bird_classify")(dialect_classify_hidden)

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
# with open("model_metadata.json", "w") as f:
#     json.dump({
#         "class_names": class_names,
#         "sample_rate": SAMPLE_RATE,
#         "sample_seconds": SAMPLE_SECONDS
#     }, f)
