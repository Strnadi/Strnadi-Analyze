import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

NUM_COLORS = 256
LUT = tf.constant(
    plt.get_cmap("magma", NUM_COLORS)(np.arange(NUM_COLORS))[:, :3].astype("float32"),
    dtype=tf.float32
)

def mel_to_magma(t):
    # t: (B,T,F) or (B,T,F,1)  →  (B,T,F,3)
    if t.shape.rank == 4 and t.shape[-1] == 1:
        t = tf.squeeze(t, -1) # (B,T,F)

    t_min = tf.reduce_min(t, axis=[1, 2], keepdims=True)
    t_max = tf.reduce_max(t, axis=[1, 2], keepdims=True)
    t_norm = (t - t_min) / (t_max - t_min + 1e-6)  # [0,1]

    idx = tf.cast(tf.round(t_norm * (NUM_COLORS - 1)), tf.int32)
    return tf.gather(LUT, idx)  # (B,T,F,3)
