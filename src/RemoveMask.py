import tensorflow as tf
import keras

#@tf.function
@keras.saving.register_keras_serializable(package="Custom")
class RemoveMask(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.identity(inputs)

    def compute_mask(self, inputs, mask=None):
        # Remove mask by returning None
        return None

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        # nothing to store, but required for serialization
        base_config = super().get_config()
        return base_config