import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa

SAMPLE_RATE = 48000
SAMPLE_SECONDS = 4

class MelToMagma(tf.keras.layers.Layer):
    """
    Converts a Mel-spectrogram to an RGB image (magma) and flattens it.
    """

    def __init__(self,
                 num_colors: int = 256,
                 cmap_name: str = "magma",
                 **kwargs):
        super().__init__(**kwargs)

        # Build a colour-lookup table (LUT) once, on the Python side.
        # Shape: (num_colors, 3)  – RGB only, ignore alpha.
        cmap = plt.get_cmap(cmap_name, num_colors)
        lut_np = cmap(np.arange(num_colors))[:, :3].astype("float32")

        # Save as a constant Tensor so it lives on the graph / GPU.
        self.lut = tf.constant(lut_np)            # (256, 3)
        self.num_colors = num_colors
        self.cmap_name = cmap_name

    def call(self, inputs):
        """
        inputs: (B, T, F) or (B, T, F, 1)  – floating point magnitudes.
        returns: (B, T * F * 3)  – flattened RGB image vectors.
        """

        # Ensure shape (B, T, F)
        if inputs.shape.rank == 4 and inputs.shape[-1] == 1:
            x = tf.squeeze(inputs, axis=-1)       # (B, T, F)
        else:
            x = inputs                            # (B, T, F)

        # Per-example min-max normalisation to [0, 1].
        x_min = tf.reduce_min(x, axis=[1, 2], keepdims=True)
        x_max = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)

        # Map to integer indices in [0, num_colors-1].
        idx = tf.cast(tf.round(x_norm * (self.num_colors - 1)), tf.int32)   # (B, T, F)

        # Look up RGB values; gather() broadcasts batch dims automatically.
        rgb = tf.gather(self.lut, idx)              # (B, T, F, 3)
        return rgb

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                num_colors=self.num_colors,
                cmap_name=self.cmap_name,
            )
        )
        return cfg

def load_and_normalize_audio(file_path, target_sr=SAMPLE_RATE):
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


# audio, _ = load_and_normalize_audio("/home/hroudis/Downloads/pretty1.wav")
# audio, _ = load_and_normalize_audio("/home/hroudis/Downloads/13.715_53.265_2020-07-11_2623_birdnet_mobile_3629931088_recording_77_00.wav")

def load_model(path):
    model = keras.saving.load_model(
        path,
        custom_objects={'Custom>MelToMagma': MelToMagma}
    )
    return model

# with open("good-model.tflite", "wb") as f:
#     tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
#     f.write(tflite_model)


# audio_tensor = tf.convert_to_tensor(np.asarray(audio).reshape(1, -1), dtype=tf.float32)

# pred = model.predict(audio_tensor, batch_size=1)
LABEL_NAMES = ['3S', 'BC', 'BD', 'BE', 'BhBl', 'BlBh', 'XlB', 'XsB']

# pred_percent = dict(zip(label_names, map(lambda x: f"{round(float(x), 2) * 100}%", pred.flatten())))

# print(pred_percent)

