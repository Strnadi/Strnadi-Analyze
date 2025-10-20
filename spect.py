import os
import keras
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import signal

SAMPLE_RATE = 16000
SAMPLE_SECONDS = 4

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

def reduce_noise(audio_data, noise_reduction_strength=0.2, noise_threshold=0.01):
    """
    Reduce background static noise using spectral gating
    
    Parameters:
    - audio_data: Audio signal to denoise
    - noise_reduction_strength: Strength of noise reduction (0.0-1.0)
    - noise_threshold: Threshold to identify noise floor
    
    Returns:
    - Denoised audio signal
    """
    # Get the spectrogram
    stft = librosa.stft(audio_data)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise floor (using lower percentile of magnitudes)
    noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
    
    # Create a mask for spectral gating
    mask = (magnitude - noise_floor * noise_threshold) / magnitude
    mask = np.maximum(mask, noise_reduction_strength)
    mask = np.minimum(mask, 1.0)
    
    # Apply the mask
    magnitude_filtered = magnitude * mask
    
    # Reconstruct signal
    stft_filtered = magnitude_filtered * np.exp(1j * phase)
    denoised_audio = librosa.istft(stft_filtered, length=len(audio_data))
    
    return denoised_audio

def visualize_spectrogram(audio_file=None, audio_data=None, save_path=None, apply_noise_reduction=True, 
                         noise_reduction_strength=0.2, noise_threshold=0.01):
    if audio_file is None and audio_data is None:
        raise ValueError("Either audio_file or audio_data must be provided")
    
    # Load audio if file path is provided
    if audio_file is not None:
        audio_data, _ = load_and_normalize_audio(audio_file)
        if audio_data is None:
            raise ValueError(f"Could not load audio from {audio_file}")
    
    # Apply noise reduction if requested
    if apply_noise_reduction:
        audio_data = reduce_noise(
            audio_data, 
            noise_reduction_strength=noise_reduction_strength,
            noise_threshold=noise_threshold
        )
    
    # Ensure we have a proper length for the model
    target_length = SAMPLE_RATE * SAMPLE_SECONDS
    if len(audio_data) < target_length:
        padded_audio = np.zeros(target_length)
        padded_audio[:len(audio_data)] = audio_data
        audio_data = padded_audio
    elif len(audio_data) > target_length:
        audio_data = audio_data[:target_length]

    audio_tensor = tf.convert_to_tensor(audio_data.reshape(1, -1), dtype=tf.float32)
    
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

    spectrogram_layer = keras.layers.MelSpectrogram(
        min_freq=4400,
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
    spectrogram = spectrogram_layer(audio_tensor)

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
    reshaped_spectrogram = keras.layers.Reshape((*spectrogram.shape[1:], 1))(spectrogram)

    # Convert single-channel spectrogram to 3 channels for ResNet
    three_channel_spectrogram = keras.layers.Concatenate(axis=-1)(
        [reshaped_spectrogram, reshaped_spectrogram, reshaped_spectrogram]
    )


    # pull out the numpy array, drop batch dim and pick channel 0
    spectrogram = three_channel_spectrogram.numpy()[0, :, :, 0]                     # shape (128, time_steps)
    # Plot spectrogram
    fig, ax = plt.subplots(figsize=(12, 6))
    img = ax.imshow(spectrogram, aspect='auto', origin='lower', cmap='magma')
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram')
    ax.set_ylabel('Mel Bins')
    ax.set_xlabel('Time Frames')
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.tight_layout()
    plt.show()
    return fig

# Update the example usage to demonstrate noise reduction
for file in os.listdir("/home/hroudis/strnad-data/BC/"):
    if file.endswith(".wav"):
        # With noise reduction (default)
        visualize_spectrogram(
            os.path.join("/home/hroudis/strnad-data/BC/", file),
            apply_noise_reduction=True, 
            noise_reduction_strength=1,  # Adjust between 0.0-1.0 (higher = more aggressive)
            noise_threshold=0.01           # Adjust to control noise detection sensitivity
        )
        
        # Optional: Uncomment to compare with no noise reduction
        # visualize_spectrogram(os.path.join("/home/hroudis/strnad-data/BC/", file), apply_noise_reduction=False)
