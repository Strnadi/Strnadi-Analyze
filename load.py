import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa

SAMPLE_RATE = 48000
SAMPLE_SECONDS = 4

class SlaneyNormalization(keras.layers.Layer):
    """
    Match librosa's Slaney filter-bank normalisation.
    Pass the number of mel bins created by MelSpectrogram.
    """
    def __init__(self, num_mel_bins, **kwargs):
        super().__init__(**kwargs)
        # Pre-compute 2 / (high-freq – low-freq) factor used by librosa
        self.scale = self.add_weight(
            name="slaney_norm",
            shape=(num_mel_bins, 1),   # one factor per mel filter
            initializer=tf.keras.initializers.Constant(2.0),  # librosa constant
            trainable=False,
        )

    def call(self, S):
        # S has shape [batch, freq, time]
        return S * self.scale   # broadcast along batch & time

class PowerToDB(keras.layers.Layer):
    """
    Convert power/amplitude spectrogram to dB exactly like
    librosa.power_to_db(S, ref=np.max, top_db=80).
    Works on [batch, freq, time] tensors.
    """
    def __init__(self, amin=1e-10, top_db=80.0, **kwargs):
        super().__init__(**kwargs)
        self.amin   = tf.constant(amin, dtype=tf.float32)
        self.top_db = tf.constant(top_db, dtype=tf.float32)

    def call(self, S):
        S = tf.maximum(S, self.amin)
        # 10*log10(S)
        log_spec = 10.0 * tf.math.log(S) / tf.math.log(tf.constant(10.0))
        # subtract per-example max → 0 dB peak
        max_per_example = tf.reduce_max(log_spec, axis=[1, 2], keepdims=True)
        log_spec = log_spec - max_per_example
        # apply top-dB floor
        log_spec = tf.maximum(log_spec, -self.top_db)
        return log_spec

class SpectralNoiseReducer(keras.layers.Layer):
    """Reduces background noise from spectrograms using spectral subtraction with smoothing.
    
    This layer estimates noise profile from low-energy frames and subtracts it from
    the spectrogram with optional smoothing and masking.
    """
    def __init__(self, 
                 noise_frame_count=10,  # Number of frames to use for noise profile
                 reduction_factor=1.5,  # How aggressively to reduce noise
                 smoothing=0.2,         # Time smoothing factor (0-1)
                 min_gain_db=-15.0,     # Minimum allowed reduction in dB 
                 freq_axis=1,           # Spectrogram frequency axis
                 time_axis=2,           # Spectrogram time axis
                 epsilon=1e-10,         # Small value to prevent division by zero
                 **kwargs):
        super().__init__(**kwargs)
        self.noise_frame_count = noise_frame_count
        self.reduction_factor = reduction_factor
        self.smoothing = smoothing
        self.min_gain_db = min_gain_db
        self.freq_axis = freq_axis
        self.time_axis = time_axis
        self.epsilon = epsilon
        self.noise_profile = None
    
    def build(self, input_shape):
        # Initialize noise profile tensor
        self.noise_profile = self.add_weight(
            name="noise_profile",
            shape=(input_shape[self.freq_axis],),
            initializer="zeros",
            trainable=False,
        )
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        # Calculate frame energy across frequency bins
        frame_energy = tf.reduce_mean(inputs, axis=self.freq_axis)
        
        # Find the indices of the lowest-energy frames
        _, indices = tf.nn.top_k(-frame_energy, k=self.noise_frame_count)
        
        # Estimate noise profile from the lowest energy frames
        noise_frames = tf.gather(inputs, indices, axis=self.time_axis)
        estimated_noise = tf.reduce_mean(noise_frames, axis=self.time_axis)
        
        # Update noise profile with smoothing
        if training:
            # Only update noise profile during training
            self.noise_profile.assign(
                self.noise_profile * (1 - self.smoothing) + 
                estimated_noise * self.smoothing
            )
        
        # Apply spectral subtraction with flooring
        # Fix: Reshape noise_profile to match input dimensions properly
        noise_profile_adjusted = self.noise_profile * self.reduction_factor
        
        # Add necessary dimensions to match the input shape
        # For input shape [batch, freq, time], we need to add a time dimension
        noise_adjusted = tf.reshape(
            noise_profile_adjusted,
            [1, tf.shape(noise_profile_adjusted)[0], 1]
        )
        
        # Ensure we don't reduce by more than min_gain_db
        min_gain = tf.pow(10.0, self.min_gain_db / 10.0)
        gain = tf.maximum(
            (inputs - noise_adjusted) / (inputs + self.epsilon),
            min_gain
        )
        
        # Apply the gain to the input spectrogram
        return inputs * gain
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "noise_frame_count": self.noise_frame_count,
            "reduction_factor": self.reduction_factor,
            "smoothing": self.smoothing,
            "min_gain_db": self.min_gain_db,
            "freq_axis": self.freq_axis,
            "time_axis": self.time_axis,
            "epsilon": self.epsilon,
        })
        return config

class SpectrogramNoiseReducer(tf.keras.layers.Layer):
    """
    Differentiable noise-reduction layer for magnitude spectrograms.

    Args
    ----
    n_std_thresh:     # ×σ added to the noise floor to build the threshold
    noise_frames:     # how many initial time frames are treated as “pure noise”
    smoothing_bins:   # vertical smoothing of the binary mask (freq axis)
    smoothing_frames: # horizontal smoothing of the binary mask (time axis)
    prop_decrease:    # 0 → no attenuation, 1 → full attenuation below threshold
    """

    def __init__(
        self,
        n_std_thresh: float = 1.5,
        noise_frames: int = 6,
        smoothing_bins: int = 3,
        smoothing_frames: int = 5,
        prop_decrease: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_std_thresh = n_std_thresh
        self.noise_frames = noise_frames
        self.smoothing_bins = smoothing_bins
        self.smoothing_frames = smoothing_frames
        self.prop_decrease = prop_decrease

        # 2-D averaging kernel for mask smoothing (depth-wise conv2d)
        k = tf.ones(
            shape=(self.smoothing_bins, self.smoothing_frames, 1, 1),
            dtype=tf.float32,
        ) / float(self.smoothing_bins * self.smoothing_frames)
        self.kernel = tf.constant(k, dtype=tf.float32)

    def call(self, spec):  # spec shape: (B, F, T)
        spec = tf.convert_to_tensor(spec, dtype=self.compute_dtype)
        eps = tf.cast(1e-8, spec.dtype)

        # 1) Estimate noise statistics on the first `noise_frames`
        noise_patch = spec[:, :, : self.noise_frames]            # (B, F, noise_frames)
        noise_mean = tf.reduce_mean(noise_patch, axis=-1, keepdims=True)
        noise_std = tf.math.reduce_std(noise_patch, axis=-1, keepdims=True)
        thresh = noise_mean + self.n_std_thresh * noise_std      # (B, F, 1)

        # 2) Hard binary mask
        hard_mask = tf.cast(spec > thresh, spec.dtype)           # (B, F, T)

        # 3) Smooth mask with depth-wise 2-D conv
        mask4d = tf.expand_dims(tf.transpose(hard_mask, (0, 2, 1)), -1)  # (B, T, F, 1)
        smooth = tf.nn.depthwise_conv2d(mask4d, self.kernel, strides=[1, 1, 1, 1], padding="SAME")
        smooth = tf.transpose(tf.squeeze(smooth, -1), (0, 2, 1))         # back to (B, F, T)
        smooth = tf.clip_by_value(smooth, 0.0, 1.0)

        # 4) Continuous gain mask
        gain = 1.0 - self.prop_decrease * (1.0 - smooth)          # (B, F, T)

        # 5) Apply mask
        return spec * gain + eps

    # Make the layer serialisable
    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            dict(
                n_std_thresh=self.n_std_thresh,
                noise_frames=self.noise_frames,
                smoothing_bins=self.smoothing_bins,
                smoothing_frames=self.smoothing_frames,
                prop_decrease=self.prop_decrease,
            )
        )
        return cfg

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

audio, _ = load_and_normalize_audio("/home/hroudis/Downloads/pretty1.wav")
# audio, _ = load_and_normalize_audio("/home/hroudis/Downloads/13.715_53.265_2020-07-11_2623_birdnet_mobile_3629931088_recording_77_00.wav")

model = keras.saving.load_model(
    "/home/hroudis/Downloads/zdenda-checkpoint(2).keras",
    custom_objects={'Custom>MelToMagma': MelToMagma}
)

# with open("good-model.tflite", "wb") as f:
#     tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
#     f.write(tflite_model)


audio_tensor = tf.convert_to_tensor(np.asarray(audio).reshape(1, -1), dtype=tf.float32)

pred = model.predict(audio_tensor, batch_size=1)
label_names = ['3S', 'BC', 'BD', 'BE', 'BhBl', 'BlBh', 'XlB', 'XsB']

pred_percent = dict(zip(label_names, map(lambda x: f"{round(float(x), 2) * 100}%", pred.flatten())))

print(pred_percent)

