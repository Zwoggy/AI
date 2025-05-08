import keras
import tensorflow as tf
import keras_hub

from ai_functionality_old import get_weighted_loss_masked_
from src.masked_metrics import MaskedAUC, masked_precision, masked_recall, masked_f1_score

from transformers import  TFEsmForTokenClassification

def create_fusion_model_function(embed_dim, ff_dim, length_of_longest_context, maxlen, new_weights, num_decoder_blocks,
                                 num_heads, num_transformer_blocks, old, rate, voc_size):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)

    encoder_inputs = keras.layers.Input(shape=(length_of_longest_context,), name='encoder_inputs')
    cnn_inputs = keras.layers.Input(shape=(4700, 3), name='decoder_inputs')
    reshaped = keras.layers.Reshape((100, 47, 3))(cnn_inputs)

    # CNN for structural Input
    cnn_output = tf.keras.Sequential([
        keras.layers.Conv2D(16, kernel_size=(3, 3), padding="same", activation="relu"),
        #keras.layers.AveragePooling2D(pool_size=2),
        keras.layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dense(embed_dim, activation="sigmoid"),  # Align dimension
    ])(reshaped)

    y = keras.layers.RepeatVector(length_of_longest_context)(cnn_output)
    # Instanziiere das Layer mit den Gewichtungen
    if old:
        embedding_layer = keras_hub.layers.TokenAndPositionEmbedding(voc_size,
                                                                     maxlen,
                                                                     embed_dim,
                                                                     #mask_zero=True
                                                                     )
        #embedding_layer = TokenAndPositionEmbedding(maxlen, voc_size, embed_dim) ## tf_keras version
        x = embedding_layer(encoder_inputs)
        mask = embedding_layer.compute_mask(encoder_inputs)
        output_dimension = x.shape[2]
    else:
        esm_model = TFEsmForTokenClassification.from_pretrained("facebook/esm2_t36_3B_UR50D")
        outputs = esm_model(encoder_inputs, output_hidden_states=True)
        x = outputs.hidden_states[-1]
        mask = tf.cast(encoder_inputs != 0, tf.bool)  # falls Padding-ID = 0
        output_dimension = x.shape[2]
    # Encoder-Transformer (optional, wenn nicht direkt ESM2-Output genutzt wird)


    for i in range(num_transformer_blocks):
        x = keras_hub.layers.TransformerEncoder(
            intermediate_dim=output_dimension,
            num_heads=num_heads,
            dropout=rate,
        )(x, padding_mask=mask)
    encoder_outputs = keras.layers.Dense(embed_dim, activation='sigmoid')(x)
    # Decoder
    decoder_outputs = encoder_outputs
    for i in range(num_decoder_blocks):
        decoder_outputs = keras_hub.layers.TransformerDecoder(
            intermediate_dim=output_dimension,
            num_heads=num_heads,
            dropout=rate
        )(decoder_outputs,
          encoder_outputs,
          encoder_padding_mask=mask,
          decoder_padding_mask=mask
          )





    # Fusion
    fused = keras.layers.Concatenate(axis=-1)([decoder_outputs, y])


    # Fusion block to fuse structural and sequential information together
    decoder_outputs = keras.layers.Dropout(rate)(fused)
    decoder_outputs = keras.layers.Dense(12, activation='relu', name='Not_the_last_Sigmoid')(decoder_outputs)

    decoder_outputs = keras.layers.Lambda(lambda x: tf.identity(x))(decoder_outputs) # removes mask for timedistributed layer since it cant deal with a mask

    decoder_outputs_final = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid', name='Final_Sigmoid'))(
        decoder_outputs)
    model = keras.Model(inputs=[encoder_inputs, cnn_inputs], outputs=decoder_outputs_final)
    model.compile(
        optimizer=optimizer,
        loss=get_weighted_loss_masked_(new_weights),
        metrics=[#masked_accuracy,
            MaskedAUC(),
            masked_precision,
            masked_recall,
            masked_f1_score]
    )
    return model



def create_fusion_model_function_02(embed_dim, ff_dim, length_of_longest_context, maxlen, new_weights, num_decoder_blocks,
                                 num_heads, num_transformer_blocks, old, rate, voc_size):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001)

    encoder_inputs = keras.layers.Input(shape=(length_of_longest_context,), name='encoder_inputs')
    cnn_inputs = keras.layers.Input(shape=(4700, 3), name='decoder_inputs')
    reshaped = keras.layers.Reshape((100, 47, 3))(cnn_inputs)

    # CNN for structural Input
    cnn_output = tf.keras.Sequential([
        keras.layers.Conv1D(16, kernel_size=3, padding="same", activation="relu"),
        #keras.layers.AveragePooling2D(pool_size=2),
        keras.layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(embed_dim, activation="sigmoid"),  # Align dimension
    ])(cnn_inputs)

    y = keras.layers.RepeatVector(length_of_longest_context)(cnn_output)
    # Instanziiere das Layer mit den Gewichtungen
    if old:
        embedding_layer = keras_hub.layers.TokenAndPositionEmbedding(voc_size,
                                                                     maxlen,
                                                                     embed_dim,
                                                                     #mask_zero=True
                                                                     )
        #embedding_layer = TokenAndPositionEmbedding(maxlen, voc_size, embed_dim) ## tf_keras version
        x = embedding_layer(encoder_inputs)
        mask = embedding_layer.compute_mask(encoder_inputs)
        output_dimension = x.shape[2]
    else:
        esm_model = TFEsmForTokenClassification.from_pretrained("facebook/esm2_t36_3B_UR50D")
        outputs = esm_model(encoder_inputs, output_hidden_states=True)
        x = outputs.hidden_states[-1]
        mask = tf.cast(encoder_inputs != 0, tf.bool)  # falls Padding-ID = 0
        output_dimension = x.shape[2]
    # Encoder-Transformer (optional, wenn nicht direkt ESM2-Output genutzt wird)


    for i in range(num_transformer_blocks):
        x = keras_hub.layers.TransformerEncoder(
            intermediate_dim=output_dimension,
            num_heads=num_heads,
            dropout=rate,
        )(x, padding_mask=mask)
    encoder_outputs = keras.layers.Dense(embed_dim, activation='sigmoid')(x)
    # Decoder

    decoder_outputs = keras_hub.layers.TransformerDecoder(
        intermediate_dim=output_dimension,
        num_heads=num_heads,
        dropout=rate
    )(y,
      encoder_outputs,
      encoder_padding_mask=mask,
      decoder_padding_mask=mask
      )

    decoder_outputs = keras_hub.layers.TransformerDecoder(
        intermediate_dim=output_dimension,
        num_heads=num_heads,
        dropout=rate
    )(decoder_outputs,
      y,
      )





    # Fusion
    fused = keras.layers.Concatenate(axis=-1)([decoder_outputs, y])


    # Fusion block to fuse structural and sequential information together
    decoder_outputs = keras.layers.Dropout(rate)(decoder_outputs)
    decoder_outputs = keras.layers.Dense(12, activation='relu', name='Not_the_last_Sigmoid')(decoder_outputs)

    decoder_outputs = keras.layers.Lambda(lambda x: tf.identity(x))(decoder_outputs) # removes mask for timedistributed layer since it cant deal with a mask

    decoder_outputs_final = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid', name='Final_Sigmoid'))(
        decoder_outputs)
    model = keras.Model(inputs=[encoder_inputs, cnn_inputs], outputs=decoder_outputs_final)
    model.compile(
        optimizer=optimizer,
        loss=get_weighted_loss_masked_(new_weights),
        metrics=[#masked_accuracy,
            MaskedAUC(),
            masked_precision,
            masked_recall,
            masked_f1_score]
    )
    return model
