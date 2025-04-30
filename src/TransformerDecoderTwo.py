import tensorflow as tf
import tf_keras
from tf_keras import layers


class TransformerDecoderTwo(tf_keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, rate = 0.3, **kwargs):
        super(TransformerDecoderTwo, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.rate = rate

    def compute_mask(self, inputs, mask = None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    @tf.function
    def call(self, encoder_outputs, training = True, mask = None):

        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype = "int32")
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(
            query = encoder_outputs,
            value = encoder_outputs,
            key = encoder_outputs,
            attention_mask = padding_mask,
        )

        if training:
            attention_output_1 = self.dropout1(attention_output_1, training = training)

        out_1 = self.layernorm_1(encoder_outputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query = out_1,
            value = encoder_outputs,
            key = encoder_outputs,
            # attention_mask=mask,
            attention_mask = padding_mask
        )

        if training:
            attention_output_2 = self.dropout2(attention_output_2, training = training)

        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def build(self, input_shape):
        self.attention_1 = layers.MultiHeadAttention(
            num_heads = self.num_heads, key_dim = self.embed_dim, dropout = 0.3, name = "att1",
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads = self.num_heads, key_dim = self.embed_dim, dropout = 0.3, name = "att2",
        )
        self.dense_proj = tf_keras.Sequential(
            [layers.TimeDistributed(layers.Dense(self.latent_dim, activation = "relu")),
             layers.TimeDistributed(layers.Dropout(rate = self.rate)),
             layers.TimeDistributed(layers.Dense(self.embed_dim)), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

    def get_config(self):
        config = super(TransformerDecoderTwo, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'latent_dim': self.latent_dim,
            'num_heads': self.num_heads,
            'rate': self.rate

        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
