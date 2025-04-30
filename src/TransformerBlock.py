import tensorflow as tf
import tf_keras
from tf_keras import layers


class TransformerBlock(tf_keras.layers.Layer):

    def __init__(self, embed_dim = 256, num_heads = 4, ff_dim = 32, rate = 0.3, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    @tf.function
    def call(self, inputs, training = True, mask = None):
        if mask is not None:
            # Transform boolean mask to attention mask for MultiHeadAttention
            # Shape: (batch_size, seq_len) → (batch_size, 1, 1, seq_len)
            attention_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)

        else:
            attention_mask = None

        attn_output = self.att(inputs, inputs, attention_mask = attention_mask)
        if training:
            attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        if training:
            ffn_output = self.dropout2(ffn_output, training = training)
        return self.layernorm2(out1 + ffn_output)

    def compute_mask(self, inputs, mask = None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(num_heads = self.num_heads, key_dim = self.embed_dim, dropout = 0.3)
        self.ffn = tf_keras.Sequential(
            [layers.TimeDistributed(layers.Dense(self.ff_dim, activation = "relu")),
             layers.TimeDistributed(layers.Dropout(rate = self.rate)),
             layers.TimeDistributed(layers.Dense(self.embed_dim)), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon = 1e-6)

        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

    def get_config(self):

        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
