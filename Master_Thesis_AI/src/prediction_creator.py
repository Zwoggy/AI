from transformers import AutoTokenizer, TFEsmForTokenClassification, TFEsmModel

from keras_hub.layers import TransformerEncoder, TokenAndPositionEmbedding, TransformerDecoder

from tensorflow.keras import backend as K

import keras
import tensorflow as tf

from Master_Thesis_AI.src.validate_on_29_external import return_29_external_dataset_X_y


def use_model_and_predict_ma():
    """Enter a sequence to use for prediction and generate the heatmap output.
    All path need to be changed to wherever the files are stored on your computer."""
    tf.keras.backend.clear_session()
    """change the following path to the final_AI folder path"""
    model = keras.saving.load_model('./Master_Thesis_AI/models/20250917_110325_best_model_fold_no_k_fold.keras',
                       custom_objects = {'TransformerBlock': TransformerEncoder,
                                         'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                         'TransformerDecoder': TransformerDecoder, "weighted_loss": get_weighted_loss},
                       compile = False
                       )

    """change the following path to path/final_AI_weights """
    #model.load_weights('./AI/EMS2_AI/AI_weights')
    model.compile()
    tf.keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
                              to_file = './testpicture.png', show_layer_activations = True)
    print(model.summary(expand_nested = True))

    x_comb, padded_epitope_list = return_29_external_dataset_X_y(model, use_structure = True)

    predictions = model.predict(x_comb)

    # create_better_heatmap(pred_list, sequence, sequence_list_for_further_stuff)
    # reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
            axis = -1)

    return weighted_loss

# def sequence_to_text(sequence):
#     return " ".join([reverse_word_map.get(i, "") for i in sequence])
