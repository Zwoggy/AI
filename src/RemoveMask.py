import tensorflow as tf

#@tf.function
class RemoveMask(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.identity(inputs)

    def compute_mask(self, inputs, mask=None):
        # Remove mask by returning None
        return None

    @classmethod
    def from_config(cls, config):
        return cls(**config)