import keras
import tensorflow as tf

# class AttentiveStatsPool(keras.layers.Layer):
#     def build(self, shape):
#         self.w = self.add_weight(shape=(shape[-1], 1), initializer="glorot_uniform")
# 
#     def call(self, t):
#         # t: (B, T, F, C)  ➜ flatten freq
#         x = tf.reshape(t, (tf.shape(t)[0], -1, t.shape[-1]))   # (B, T, C)
#         alpha = tf.nn.softmax(tf.squeeze(tf.matmul(x, self.w), -1))# (B, T)
#         mu = tf.reduce_sum(x * alpha[..., None], axis=1)
#         sigma = tf.sqrt(tf.reduce_sum(alpha[..., None] * tf.square(x-mu[:,None,:]), axis=1)+1e-9)
#         return tf.concat([mu, sigma], axis=-1)                 # (B, 2C)

class AttentiveStatsPool(keras.layers.Layer):
    def build(self, shape):
        # Weights for the attention mechanism
        self.w = self.add_weight(shape=(shape[-1], 1), initializer="glorot_uniform")

    def call(self, t):
        # Input 't' from DenseNet: (Batch, Time, Freq, Channels)
        # We flatten Time and Freq together to create a 1D sequence of spatial patches.
        # This allows the model to attend to specific 2D locations in the feature map.
        shape = tf.shape(t)

        # x shape becomes (Batch, N, Channels), where N = Time * Freq
        x = tf.reshape(t, (shape[0], -1, t.shape[-1]))   

        # Calculate attention scores (alpha) for each spatial patch
        # Result of matmul is (B, N, 1), squeeze makes it (B, N)
        dot_product = tf.matmul(x, self.w)
        alpha = tf.nn.softmax(tf.squeeze(dot_product, axis=-1)) 

        # Expand dimensions for mathematically safe broadcasting (TFLite friendly)
        alpha_expanded = tf.expand_dims(alpha, axis=-1)  # (B, N, 1)

        # Calculate Mu (weighted mean across all spatial patches)
        mu = tf.reduce_sum(x * alpha_expanded, axis=1)   # (B, C)

        # Expand dimensions for variance calculation
        mu_expanded = tf.expand_dims(mu, axis=1)         # (B, 1, C)

        # Calculate Sigma (weighted standard deviation across all spatial patches)
        squared_diff = tf.square(x - mu_expanded)
        weighted_squared_diff = tf.reduce_sum(alpha_expanded * squared_diff, axis=1)
        sigma = tf.sqrt(weighted_squared_diff + 1e-9)    # (B, C)

        # Concatenate mean and standard deviation
        return tf.concat([mu, sigma], axis=-1)           # (B, 2C)