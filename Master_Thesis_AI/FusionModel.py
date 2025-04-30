"""
This Skripts sole purpose is to create a model that can be used to predict whether each aminoacid residue is an epitope or not.
Main Skript for the Master's Thesis.
"""
import tensorflow as tf
import tf_keras
from tensorflow.keras import layers

from ai_functionality_old import get_weighted_loss, \
    calculating_class_weights
from src.TransformerDecoderTwo import TransformerDecoderTwo
from src.TokenAndPositionEmbedding import TokenAndPositionEmbedding
from src.TransformerBlock import TransformerBlock


class FusionModel(tf_keras.Model):
    def __init__(self, length_of_longest_context, voc_size, embed_dim, ff_dim, num_heads,
                 num_transformer_encoder_blocks, num_decoder_blocks, rate=0.3, training=True, **kwargs):
        super(FusionModel, self).__init__(**kwargs)
        
        ### Initialize variables
        self.num_transformer_encoder_blocks = num_transformer_encoder_blocks
        self.length_of_longest_context = length_of_longest_context
        self.voc_size = voc_size
        self.training = training
        self.num_decoder_blocks = num_decoder_blocks
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.rate = rate

        
    @tf.function
    def call(self, inputs, training=False, mask=None):
        encoder_inputs, structure_input = inputs
        # Input and embedding
        encoder_embed_out = self.embedding_layer(encoder_inputs)
        x = encoder_embed_out
        output_dimension = x.shape[2]


        for block in self.transformer_blocks:
            x = block(x, training=training)

        for decoder in self.decoder_layers:
            x = decoder(x, training=training)

        # CNN Strukturpfad
        y = self.cnn(structure_input)
        y = layers.RepeatVector(self.length_of_longest_context)(y)

        # Fusion
        fused = layers.Concatenate(axis=-1)([x, y])
        fused = self.fusion_dense(fused)
        fused = self.fusion_norm(fused)

        # Decoder
        fusion_output = fused


        fusion_output = layers.Dropout(self.rate)(fusion_output)
        fusion_output = self.output_dense1(fusion_output)
        return self.output_dense2(fusion_output)

    def build(self, input_shape):
        # Embedding Layer
        self.embedding_layer = TokenAndPositionEmbedding(self.length_of_longest_context, self.voc_size, self.embed_dim)
        # CNN Pfad
        self.cnn = tf_keras.Sequential([
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(self.embed_dim, activation="relu"),  # Align dimension
        ])

        # Transformer Blocks
        self.transformer_blocks = [TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.rate)
                                   for _ in range(self.num_transformer_encoder_blocks)]

        # Fusion
        self.fusion_dense = layers.Dense(self.embed_dim, activation="relu")
        self.fusion_norm = layers.LayerNormalization()

        # Decoder
        self.decoder_layers = [TransformerDecoderTwo(self.embed_dim, self.ff_dim, self.num_heads)
                               for _ in range(self.num_decoder_blocks)]

        self.output_dense1 = layers.Dense(12, activation="sigmoid", name='Not_the_last_Sigmoid')
        self.output_dense2 = layers.TimeDistributed(
            layers.Dense(1, activation="sigmoid", name='Final_Sigmoid'))

    def get_config(self):
        config = super(FusionModel, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'length_of_longest_context': self.length_of_longest_context,
            'voc_size': self.voc_size,
            'num_transformer_encoder_blocks': self.num_transformer_encoder_blocks,
            'num_decoder_blocks': self.num_decoder_blocks
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)




def create_model():

    # Initialize weights
    new_weights = calculating_class_weights(epitope_list)

    model = FusionModel(
        length_of_longest_context=235,
        voc_size=28,
        embed_dim=24, # 512?
        ff_dim=32, #2048
        num_heads=40,
        num_transformer_encoder_blocks=2,
        num_decoder_blocks=2,
        rate=0.3,
        training=True
    )

    # Kompilieren
    #model = tf_keras.Model(inputs=encoder_inputs, outputs=decoder_outputs_final)

    model.compile(optimizer, loss=get_weighted_loss(new_weights),
                  weighted_metrics=['accuracy', tf_keras.metrics.AUC(), tf_keras.metrics.Precision(),
                                    tf_keras.metrics.Recall()])
    # model.compile(optimizer, loss="binary_crossentropy", weighted_metrics=['accuracy', tf.keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])
    print("training_data:", training_data[0])  # debug
    history = model.fit(x=training_data, y=epitope_list, batch_size=16, epochs=100,
                        validation_data=(testx_list, testy_list), callbacks=[callback], verbose=1)

    # model.fit({"encoder_inputs": encoder_data, "structure_input": structure_data}, y_true, ...)










if __name__=="__main__":
    pass