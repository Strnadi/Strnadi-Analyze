import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import signal
from functools import cache
import scipy, scipy.io, scipy.signal
import noisereduce as nr
import librosa

SAMPLE_RATE = 48000
SAMPLE_SECONDS = 5

class NoiseReduceLayer(keras.layers.Layer):
    """
    Wraps noisereduce.reduce_noise for fixed-length 1D audio inputs.
    NOTE: uses tf.numpy_function -> no gradients, limited serialization/portability.
    """
    def __init__(self, sr=16000, n_fft=2048, prop_decrease=1.0, **kwargs):
        super().__init__(**kwargs)
        self.sr = sr
        self.n_fft = n_fft
        self.prop_decrease = prop_decrease
        # This layer doesn't have trainable weights:
        self.trainable = False

    def _reduce_np(self, y_np):
        # y_np: 1D numpy array (float32 or float64)
        den = nr.reduce_noise(
            y=y_np,
            sr=self.sr,
            n_fft=self.n_fft,
            prop_decrease=self.prop_decrease
        )
        # ensure dtype matches TF dtype
        return den.astype(np.float32)

    def call(self, inputs, training=None):
        """
        inputs: Tensor shape (batch, time) or (time,) for single example.
        Returns tensor same shape as inputs with dtype tf.float32.
        """
        # handle single example (rank 1)
        if tf.rank(inputs) == 1:
            out = tf.numpy_function(self._reduce_np, [inputs], tf.float32)
            out.set_shape(inputs.shape)
            return out

        # handle batch: map the numpy_function across the batch
        def apply_one(x):
            y = tf.numpy_function(self._reduce_np, [x], tf.float32)
            # numpy_function loses shape info: restore it
            y.set_shape(x.shape)
            return y

        out = tf.map_fn(apply_one, inputs, dtype=tf.float32)
        # set full shape (batch_dim, time_dim) if known statically
        out.set_shape(inputs.shape)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

class StaticNoiseReducer(keras.layers.Layer):
    def __init__(self, percentile=0.1, floor_margin_db=5.0, reduction_axis=1, epsilon=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.percentile = percentile
        self.floor_margin_db = floor_margin_db
        self.reduction_axis = reduction_axis
        self.epsilon = epsilon

    def call(self, inputs):
        sorted_vals = tf.sort(inputs, axis=self.reduction_axis)
        axis_len = tf.shape(sorted_vals)[self.reduction_axis]
        idx = tf.cast(tf.round(self.percentile * tf.cast(axis_len - 1, tf.float32)), tf.int32)
        noise_floor = tf.gather(sorted_vals, idx, axis=self.reduction_axis)
        noise_floor = tf.expand_dims(noise_floor, axis=self.reduction_axis)
        threshold = noise_floor + self.floor_margin_db
        return tf.maximum(inputs - threshold, self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "percentile": self.percentile,
                "floor_margin_db": self.floor_margin_db,
                "reduction_axis": self.reduction_axis,
                "epsilon": self.epsilon,
            }
        )
        return config

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

class LibrosaPowerToDb(keras.layers.Layer):
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

# Custom audio loading and processing functions
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

def visualize_spectrogram(audio_file=None, audio_data=None, save_path=None, apply_noise_reduction=False, 
                         noise_reduction_strength=0.2, noise_threshold=0.01):
    if audio_file is None and audio_data is None:
        raise ValueError("Either audio_file or audio_data must be provided")
    
    # Load audio if file path is provided
    if audio_file is not None:
        audio_data, _ = load_and_normalize_audio(audio_file)
        if audio_data is None:
            raise ValueError(f"Could not load audio from {audio_file}")

    audio_tensor = tf.convert_to_tensor(np.asarray(audio_data).reshape(1, -1), dtype=tf.float32)

    # Create a spectrogram manually using the same parameters
    # mel = keras.layers.MelSpectrogram(
    #     min_freq=4000,
    #     max_freq=9000,
    #     num_mel_bins=512,
    #     sequence_length=1024,
    #     sequence_stride=128,
    #     fft_length=4 * SAMPLE_RATE,
    #     sampling_rate=SAMPLE_RATE,
    #     power_to_db=True,
    #     # min_power=0.0,
    #     window="hann",
    #     name="mel_spectrogram"
    # )
    # x = mel(audio_tensor)
    # x = keras.layers.Reshape((*x.shape[1:], 1))(x)
    # x = keras.layers.Concatenate(axis=-1)([x, x, x])

    # spectrogram_layer = keras.layers.MelSpectrogram(
    #     min_freq=4400,
    #     max_freq=9000,
    #     num_mel_bins=512,
    #     sequence_length=1024,
    #     sequence_stride=128,
    #     fft_length=4 * SAMPLE_RATE,
    #     sampling_rate=SAMPLE_RATE,
    #     power_to_db=True,
    #     window="hann",
    #     name="mel_spectrogram"
    # )
    # spectrogram = spectrogram_layer(audio_tensor)

    spectrogram_layer = keras.layers.MelSpectrogram(
        min_freq=3500,
        max_freq=8500,
        num_mel_bins=128,
        # sequence_length=2048,
        # sequence_stride=256,
        fft_length=1024,
        sampling_rate=48000,
        # power_to_db=True,
        # max_power=0.0,
        power_to_db=False,
        mag_exp=2.0
    )

    spectrogram = spectrogram_layer(audio_tensor)

    # spectrogram = SpectrogramNoiseReducer(n_std_thresh=12, smoothing_bins=5, smoothing_frames=5)(spectrogram)
    spectrogram = SlaneyNormalization(128)(spectrogram)
    spectrogram = LibrosaPowerToDb()(spectrogram)
    spectrogram = MelToMagma()(spectrogram)

    # spectrogram = StaticNoiseReducer(percentile=0.1, floor_margin_db=5.0, name="spectral_noise_gate")(spectrogram)

    # spectrogram_denoised = SpectralNoiseReducer(
    #     noise_frame_count=15,
    #     reduction_factor=10,
    #     smoothing=0.5,
    #     min_gain_db=-20.0,
    #     name="spectral_noise_reducer"
    # )(spectrogram)

    # spectrogram_denoised = StaticNoiseReducer(percentile=0.1, floor_margin_db=20.0, name="spectral_noise_gate")(spectrogram)

    # Add a channel dimension
    # reshaped_spectrogram = keras.layers.Reshape((*spectrogram.shape[1:], 1))(spectrogram)

    # three_channel_spectrogram = reshaped_spectrogram

    # Convert single-channel spectrogram to 3 channels for ResNet
    # three_channel_spectrogram = keras.layers.Concatenate(axis=-1)(
    #     [reshaped_spectrogram, reshaped_spectrogram, reshaped_spectrogram]
    # )


    # pull out the numpy array, drop batch dim and pick channel 0
    # spectrogram = three_channel_spectrogram.numpy()[0, :, :, 0]                     # shape (128, time_steps)

    # print(spectrogram.shape, spectrogram)

    # Plot spectrogram
    # fig, ax = plt.subplots(figsize=(12, 6))
    # img = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    # plt.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set_title('Mel Spectrogram')
    # ax.set_ylabel('Mel Bins')
    # ax.set_xlabel('Time Frames')
    
    # if save_path is not None:
    #     plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    # plt.tight_layout()
    # plt.show()
    # return fig

    plt.figure(figsize=(6,4))
    plt.imshow(spectrogram.numpy()[0], origin="lower")
    # librosa.display.specshow(spectrogram, sr=48000, hop_length=512, x_axis='time', y_axis='mel', cmap='magma')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Update the example usage to demonstrate noise reduction
# for file in os.listdir("/home/hroudis/strnad-data/BC/"):
#     if file.endswith(".wav"):
#         # With noise reduction (default)
#         visualize_spectrogram(
#             os.path.join("/home/hroudis/strnad-data/BC/", file),
#             apply_noise_reduction=False, 
#             noise_reduction_strength=1,  # Adjust between 0.0-1.0 (higher = more aggressive)
#             noise_threshold=0.01           # Adjust to control noise detection sensitivity
#         )

visualize_spectrogram(
    # "/home/hroudis/Downloads/pretty.wav",
    "/home/hroudis/strnad-data/BC/-0.452_43.065_2020-07-03_1552_birdnet_mobile_3778446732_recording_0_05.wav",
    apply_noise_reduction=False, 
    noise_reduction_strength=1,  # Adjust between 0.0-1.0 (higher = more aggressive)
    noise_threshold=0.01           # Adjust to control noise detection sensitivity
)
