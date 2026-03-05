# Requirements:
# pip install -U "tf-keras-vis" matplotlib numpy tensorflow

import io
import sys
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import glob

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

import librosa
import matplotlib.pyplot as plt

LABELS = "BC,BE,BhBl,BlBh,None,Unfinished,XB".split(",")

def decode_predictions(preds, top=5):
    results = []
    for pred in preds:
        # Get the indices of the top predictions
        top_indices = np.argsort(pred)[::-1][:top]
        classes = [LABELS[idx] for idx in top_indices]
        results.append(classes)
    return results

def decode_predictions_with_probability(preds, top=5):
    results = []
    for pred in preds:
        # Get the indices of the top predictions
        top_indices = np.argsort(pred)[::-1][:top]
        class_probs = [(LABELS[idx], pred[idx]) for idx in top_indices]
        results.append(class_probs)
    return results

SAMPLE_RATE = 48000

def load_audio(file: str | io.BytesIO, target_sr=SAMPLE_RATE):
    audio, sr = librosa.load(file, sr=target_sr, mono=True)
    return audio, sr

NUM_COLORS = 256
LUT = tf.constant(
    plt.get_cmap("magma", NUM_COLORS)(np.arange(NUM_COLORS))[:, :3].astype("float32"),
    dtype=tf.float32
)

@keras.saving.register_keras_serializable()
def mel_to_magma(t):
    # t: (B,T,F) or (B,T,F,1)  →  (B,T,F,3)
    if t.shape.rank == 4 and t.shape[-1] == 1:
        t = tf.squeeze(t, -1) # (B,T,F)

    t_min = tf.reduce_min(t, axis=[1, 2], keepdims=True)
    t_max = tf.reduce_max(t, axis=[1, 2], keepdims=True)
    t_norm = (t - t_min) / (t_max - t_min + 1e-6)  # [0,1]

    idx = tf.cast(tf.round(t_norm * (NUM_COLORS - 1)), tf.int32)
    return tf.gather(LUT, idx)  # (B,T,F,3)

@keras.saving.register_keras_serializable()
class GlobalGeMPool2D(keras.layers.Layer):
    def __init__(self, p_init=3.0, **kwargs):
        super().__init__(**kwargs);
        self.p = tf.Variable(p_init, dtype=tf.float32)

    def call(self, t):
        t = tf.maximum(t, 1e-6)
        return tf.pow(tf.reduce_mean(tf.pow(t, self.p), axis=[1, 2]), 1./self.p)

@keras.saving.register_keras_serializable()
class AttentiveStatsPool(keras.layers.Layer):
    def build(self, shape):
        self.w = self.add_weight(shape=(shape[-1], 1), initializer="glorot_uniform")

    def call(self, t):
        # t: (B, T, F, C) ➜ flatten freq
        # We use tf.shape(t)[0] for the batch size to handle dynamic batch sizes safely
        shape = tf.shape(t)
        x = tf.reshape(t, (shape[0], -1, t.shape[-1]))   # (B, T, C)

        # Calculate alpha
        # Result of matmul is (B, T, 1), squeeze makes it (B, T)
        dot_product = tf.matmul(x, self.w)
        alpha = tf.nn.softmax(tf.squeeze(dot_product, axis=-1)) # (B, T)

        # FIX: Use tf.expand_dims instead of alpha[..., None]
        # This replaces the complex StridedSlice with a native TFLite ExpandDims op
        alpha_expanded = tf.expand_dims(alpha, axis=-1)  # (B, T, 1)

        # Calculate Mu
        mu = tf.reduce_sum(x * alpha_expanded, axis=1)   # (B, C)

        # FIX: Use tf.expand_dims instead of mu[:, None, :]
        mu_expanded = tf.expand_dims(mu, axis=1)         # (B, 1, C)

        # Calculate Sigma
        # (x - mu_expanded) broadcasts correctly now
        squared_diff = tf.square(x - mu_expanded)
        weighted_squared_diff = tf.reduce_sum(alpha_expanded * squared_diff, axis=1)
        sigma = tf.sqrt(weighted_squared_diff + 1e-9)

        return tf.concat([mu, sigma], axis=-1)           # (B, 2C)

model = keras.saving.load_model("good-model-2.keras", custom_objects={"mel_to_magma": mel_to_magma, "GlobalGeMPool2D": GlobalGeMPool2D, "AttentiveStatsPool": AttentiveStatsPool})
model.summary()


# Build seed input through model preprocessing so GradCAM sees DenseNet-ready features.
preprocess_model = keras.Model(inputs=model.input, outputs=model.get_layer("lambda").output)
preprocess_model.summary()


# GradCAM model starts at DenseNet image-like input.
densenet = model.get_layer("densenet121")
tail_input = keras.Input(shape=densenet.input_shape[1:], name="rgb_mel_input")

# 1. Save the new DenseNet output specifically
densenet_features = densenet(tail_input, training=False)
if isinstance(densenet_features, (list, tuple)):
    densenet_features = densenet_features[0]

# 2. Pass those features through the rest of the network
x = model.get_layer("attentive_stats_pool")(densenet_features, training=False)
x = model.get_layer("dense")(x, training=False)
x = model.get_layer("dropout")(x, training=False)
x = model.get_layer("dense_1")(x, training=False)
x = model.get_layer("dropout_1")(x, training=False)
tail_output = model.get_layer("dense_2")(x, training=False)

# 3. Use the newly created tensor (densenet_features) as the output
cam_model = keras.Model(
    inputs=tail_input, 
    outputs=[densenet_features, tail_output], 
    name="classifier_tail"
)
cam_model.summary()


for file in glob.glob("*.wav"):

    _loaded_audio, _ = load_audio(file)
    expected_samples = model.input_shape[-1]

    # Trim/pad to match model input length.
    if _loaded_audio.shape[0] > expected_samples:
        _loaded_audio = _loaded_audio[:expected_samples]
    elif _loaded_audio.shape[0] < expected_samples:
        _loaded_audio = np.pad(_loaded_audio, (0, expected_samples - _loaded_audio.shape[0]))

    _loaded_audio = _loaded_audio[np.newaxis, :]  # (B, 192000)

    seed_tensor = tf.constant(_loaded_audio, dtype=tf.float32)
    seed_input = preprocess_model(seed_tensor, training=False)
    print("Shape: ", seed_input.shape)


    # 4. Unpack the two outputs when predicting!
    # Since the model now outputs [features, predictions], we must unpack them.
    _, preds = cam_model.predict(seed_input)
    predictions = decode_predictions_with_probability(preds, top=3)[0]

    prediction_indices = [LABELS.index(prediction[0]) for prediction in predictions]
    print(predictions)
    argmax = prediction_indices[0]

    # 5. Run the GradientTape using the new cam_model
    with tf.GradientTape() as tape:
        conv_outputs, model_predictions = cam_model(seed_input, training=False)
        class_channel = model_predictions[:, argmax]

    # gradient of the class score w.r.t. conv layer output
    grads = tape.gradient(class_channel, conv_outputs)  # shape: (1, h, w, channels)

    # 4) channel-wise mean of gradients (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (channels,)

    # 5) get numpy arrays (for postprocessing / visualization)
    pooled_grads_value = pooled_grads.numpy()
    conv_layer_output_value = conv_outputs[0].numpy()  # shape: (h, w, channels)

    # 6) weight the channels by the importance (vectorized)
    conv_layer_output_value *= pooled_grads_value[np.newaxis, np.newaxis, :]
    # or if you prefer a loop (slower):
    # for i in range(conv_layer_output_value.shape[-1]):
    #     conv_layer_output_value[..., i] *= pooled_grads_value[i]

    # 7) compute the heatmap (spatially average channels and normalize)
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    import matplotlib.cm as cm

    # 1. Extract the original image-like features from the batch
    if isinstance(seed_input, tf.Tensor):
        original_img = seed_input[0].numpy()
    else:
        original_img = seed_input[0]

    # Ensure the original image is in [0, 1] range for matplotlib
    original_img = np.clip(original_img, 0.0, 1.0)

    # original_img = mel_to_magma(seed_tensor[np.newaxis, ...])[0].numpy()
    # original_img = np.clip(original_img, 0.0, 1.0)

    # 2. Resize the heatmap to match the original image dimensions
    # The heatmap is currently the size of the DenseNet output (e.g., 7x11).
    # We need it to be 224x376 to overlay correctly.
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], 
        size=(original_img.shape[0], original_img.shape[1]),
        method='bilinear'
    ).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)

    # 3. Create a side-by-side plot
    plt.figure(figsize=(14, 6))

    # Left Subplot: Original Spectrogram
    plt.subplot(1, 2, 1)
    # Add origin='lower' here
    plt.imshow(original_img, origin='lower')
    plt.title(f"Original Input\nPredicted: {LABELS[argmax]}")
    plt.axis('off')

    # Right Subplot: Grad-CAM Overlay
    plt.subplot(1, 2, 2)
    # Add origin='lower' to BOTH the base image and the heatmap overlay
    plt.imshow(original_img, origin='lower')
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5, origin='lower') 
    plt.title("Grad-CAM Heatmap Overlay")
    plt.axis('off')

    # 4. Save to disk
    output_path = f"heatmap_output_{file}_{LABELS[argmax]}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close() 

    print(f"Successfully saved heatmap visualization to: {output_path}")
