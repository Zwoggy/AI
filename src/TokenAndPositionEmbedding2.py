import tensorflow as tf
import tf_keras
from tf_keras import layers


class TokenAndPositionEmbedding2(tf_keras.layers.Layer):

    def __init__(self, maxlen = 1000, vocab_size = 100, embed_dim = 40, **kwargs):
        super(TokenAndPositionEmbedding2, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.token_emb = layers.Embedding(input_dim = self.vocab_size, output_dim = self.embed_dim, mask_zero = True)
        self.pos_emb = layers.Embedding(input_dim = self.maxlen, output_dim = self.embed_dim, mask_zero = True)

    def compute_mask(self, inputs, mask = None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    @tf.function
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start = 0, limit = maxlen, delta = 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super(TokenAndPositionEmbedding2, self).get_config()
        # config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,

        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
