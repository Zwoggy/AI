import tensorflow as tf

@tf.function
class RemoveMask(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs  # tf.identity optional

    def compute_mask(self, inputs, mask=None):
        return None  # ← das ist entscheidend!