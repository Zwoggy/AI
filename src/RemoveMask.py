import tensorflow as tf

class RemoveMask(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs  # tf.identity optional
