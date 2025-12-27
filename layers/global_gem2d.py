import keras
import tensorflow as tf

class GlobalGeMPool2D(keras.layers.Layer):
    def __init__(self, p_init=3.0):
        super().__init__();
        self.p = tf.Variable(p_init, dtype=tf.float32)

    def call(self, t):
        t = tf.maximum(t, 1e-6)
        return tf.pow(tf.reduce_mean(tf.pow(t, self.p), axis=[1, 2]), 1./self.p)
