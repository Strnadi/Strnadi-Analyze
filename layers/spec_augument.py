import keras
import tensorflow as tf


class SpecAugment(keras.Layer):
    """
    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    (Park et al., 2019 — https://arxiv.org/abs/1904.08779)

    Applies the following augmentations to a mel spectrogram **only during training**:
      - Frequency masking: zeroes out `num_freq_masks` random bands of consecutive
        mel bins, each of width up to `freq_mask_param` bins.
      - Time masking:      zeroes out `num_time_masks` random bands of consecutive
        time steps, each of width up to min(time_mask_param, T * max_time_mask_ratio)
        frames.

    Expected input shape: (Batch, Freq, Time)  — i.e. the raw output of
    keras.layers.MelSpectrogram before any transpose/reshape.

    Args:
        freq_mask_param     (int):   Maximum width of a single frequency mask (F in the paper). Default 27.
        time_mask_param     (int):   Maximum width of a single time mask (T in the paper). Default 100.
        num_freq_masks      (int):   How many independent frequency masks to apply. Default 2.
        num_time_masks      (int):   How many independent time masks to apply. Default 2.
        max_time_mask_ratio (float): Caps time-mask width to this fraction of total
                                     frames (p in the paper). Default 0.2.
        mask_value          (float): Value to fill masked regions with. Default 0.0.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        max_time_mask_ratio: float = 0.2,
        mask_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.max_time_mask_ratio = max_time_mask_ratio
        self.mask_value = mask_value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_freq_mask(self, spec: tf.Tensor) -> tf.Tensor:
        """Mask random frequency bands. spec: (B, F, T)"""
        num_freq = tf.shape(spec)[1]

        for _ in range(self.num_freq_masks):
            # Width of this mask
            f = tf.random.uniform(
                shape=(), minval=0, maxval=self.freq_mask_param + 1, dtype=tf.int32
            )
            # Starting bin (ensure f0 + f <= num_freq)
            f0 = tf.random.uniform(
                shape=(), minval=0, maxval=tf.maximum(num_freq - f, 1), dtype=tf.int32
            )

            # Build a boolean mask over the freq axis: True = keep, False = zero
            # Shape: (1, F, 1) — broadcasts over batch and time
            freq_indices = tf.range(num_freq)                        # (F,)
            mask = ~((freq_indices >= f0) & (freq_indices < f0 + f)) # (F,)
            mask = tf.reshape(tf.cast(mask, spec.dtype), [1, num_freq, 1])

            spec = spec * mask + (1.0 - mask) * self.mask_value

        return spec

    def _apply_time_mask(self, spec: tf.Tensor) -> tf.Tensor:
        """Mask random time bands. spec: (B, F, T)"""
        num_time = tf.shape(spec)[2]

        # Cap the maximum mask width by max_time_mask_ratio
        max_t = tf.cast(
            tf.minimum(
                tf.cast(self.time_mask_param, tf.float32),
                tf.cast(num_time, tf.float32) * self.max_time_mask_ratio,
            ),
            tf.int32,
        )
        max_t = tf.maximum(max_t, 1)

        for _ in range(self.num_time_masks):
            t = tf.random.uniform(
                shape=(), minval=0, maxval=max_t + 1, dtype=tf.int32
            )
            t0 = tf.random.uniform(
                shape=(), minval=0, maxval=tf.maximum(num_time - t, 1), dtype=tf.int32
            )

            # Shape: (1, 1, T) — broadcasts over batch and freq
            time_indices = tf.range(num_time)
            mask = ~((time_indices >= t0) & (time_indices < t0 + t))
            mask = tf.reshape(tf.cast(mask, spec.dtype), [1, 1, num_time])

            spec = spec * mask + (1.0 - mask) * self.mask_value

        return spec

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Args:
            inputs:   Mel spectrogram, shape (B, F, T).
            training: Augmentation is applied ONLY when training=True.
        Returns:
            Augmented (or unchanged) spectrogram, same shape as inputs.
        """
        if not training:
            return inputs

        x = self._apply_freq_mask(inputs)
        x = self._apply_time_mask(x)
        return x

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "freq_mask_param": self.freq_mask_param,
                "time_mask_param": self.time_mask_param,
                "num_freq_masks": self.num_freq_masks,
                "num_time_masks": self.num_time_masks,
                "max_time_mask_ratio": self.max_time_mask_ratio,
                "mask_value": self.mask_value,
            }
        )
        return config
