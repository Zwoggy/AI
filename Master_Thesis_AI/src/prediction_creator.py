from transformers import AutoTokenizer, TFEsmForTokenClassification, TFEsmModel

from keras_hub.layers import TransformerEncoder, TokenAndPositionEmbedding, TransformerDecoder

from tensorflow.keras import backend as K

import keras
from matplotlib import pyplot as plt
import numpy as np
import pickle
import seaborn as sb
import tensorflow as tf

from Master_Thesis_AI.src.validate_on_29_external import return_29_external_dataset_X_y
from src import RemoveMask


def use_model_and_predict_ma():
    """Enter a sequence to use for prediction and generate the heatmap output.
    All path need to be changed to wherever the files are stored on your computer."""
    tf.keras.backend.clear_session()
    """change the following path to the final_AI folder path"""
    model = keras.saving.load_model('./Master_Thesis_AI/models/20250921_121619_best_model_fold_no_k_fold.keras',
                       custom_objects = {'TransformerBlock': TransformerEncoder,
                                         'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                         'TransformerDecoder': TransformerDecoder, "weighted_loss": get_weighted_loss,
                                         'RemoveMask': RemoveMask},
                       compile = False
                       )

    """change the following path to path/final_AI_weights """
    #model.load_weights('./AI/EMS2_AI/AI_weights')
    model.compile()
    tf.keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
                              to_file = './testpicture.png', show_layer_activations = True)
    print(model.summary(expand_nested = True))

    x_comb, padded_epitope_list = return_29_external_dataset_X_y(model, maxlen=933, use_structure = True)
    # x_comb: array consisting of:
    # 29 proteins
    # 933 sequence (length of 933)
    # 8 infos (relevant: hier nur index 0)

    predictions = model.predict(x_comb)

    # für jedes protein (29x) das hier machen
    decoded_sequence = sequence_to_text(sequence)
    pred_list, sequence_list_for_further_stuff = blub(predictions, decoded_sequence)
    create_better_heatmap(pred_list, decoded_sequence, sequence_list_for_further_stuff)


def sequence_to_text(sequence):
    with open('./AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    reverse_word_map = dict(map(reversed, encoder.word_index.items()))
    return "".join([reverse_word_map.get(i, "") for i in sequence])

def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
            axis = -1)

    return weighted_loss

def create_better_heatmap(data, sequence, sequence_list):
    """Input: predictions from the model
    Output: Heatmaps according to the predictions for the whole sequence entered"""

    data = np.array(data[:len(sequence)], dtype = np.float32)
    sequence_list = np.array(sequence_list[:len(sequence)], dtype = str) ## used to be np.str

    data_list, sequence_list = create_blocks(data, sequence_list)
    print(data_list, sequence_list)
    data_list = np.reshape(data_list, (data_list.shape[1], data_list.shape[0]))
    sequence_list = np.reshape(sequence_list, (sequence_list.shape[1], sequence_list.shape[0]))
    print("------------------------------------------------")
    print(data_list.shape)

    """change the path to a folder to save the pictuers in"""
    filename = "./AI/pictures/" + "new" + ".png"

    plt.figure(dpi = 1000)
    sb.heatmap(data_list, xticklabels = False, yticklabels = False, vmin = 0.2, vmax = 0.8, cmap = "rocket_r", annot=sequence_list, fmt="")
    plt.savefig(filename, dpi = 1000, bbox_inches = "tight")
    plt.show()

def create_blocks(list1, list2):
    """creates blocks of max size 20 so that every heatmap has a max length of 20"""
    block_size = 20
    num_blocks1 = len(list1) // block_size
    num_blocks2 = len(list2) // block_size
    blocks1 = []
    blocks2 = []

    for i in range(num_blocks1):
        start = i * block_size
        end = start + block_size
        block1 = np.array(list1[start:end])
        block2 = np.array(list2[start:end])
        blocks1.append(block1)
        blocks2.append(block2)

    return np.array(blocks1), np.array(blocks2)

def blub(predictions, sequence):
    x = 1
    pred_list = []
    sequence_list_for_further_stuff = []
    for i, (pred, seq) in enumerate(zip(predictions, sequence)):
        for (j, seq2) in zip(pred, sequence):
            pred_list.append(j)
            sequence_list_for_further_stuff.append(seq2)
            print(str(x) + ": " + str(j) + " - " + str(sequence[x - 1]))
            x += 1
    return pred_list, sequence_list_for_further_stuff
