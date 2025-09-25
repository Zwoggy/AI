import pickle

import keras
import numpy as np
import seaborn as sb
import tensorflow as tf
from keras_hub.layers import TransformerEncoder, TokenAndPositionEmbedding, TransformerDecoder
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K

from Master_Thesis_AI.src.validate_on_29_external import return_29_external_dataset_X_y
from src import RemoveMask
from src.masked_metrics import masked_mcc, masked_f1_score, masked_precision, masked_recall, masked_auc


def use_model_and_predict_ma():
    """Enter a sequence to use for prediction and generate the heatmap output.
    All path need to be changed to wherever the files are stored on your computer."""
    tf.keras.backend.clear_session()
    #TODO change the following path to the final_AI folder path
    model = keras.saving.load_model('./Master_Thesis_AI/models/20250921_121619_best_model_fold_no_k_fold.keras',
                       custom_objects = {'TransformerBlock': TransformerEncoder,
                                         'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                         'TransformerDecoder': TransformerDecoder, "weighted_loss": get_weighted_loss,
                                         'RemoveMask': RemoveMask},
                       compile = False
                       )

    #TODO change the following path to path/final_AI_weights
    #model.load_weights('./AI/EMS2_AI/AI_weights')
    model.compile()
    tf.keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
                              to_file = './AI/pictures/testpicture.png', show_layer_activations = True)
    print(model.summary(expand_nested = True))

    x_comb, padded_epitope_list, id_list = return_29_external_dataset_X_y(model, maxlen=933, use_structure = True)
    # x_comb: array consisting of:
    # 29 proteins
    # 933 sequence (length of 933)
    # 8 infos (relevant: here only index 0)

    # contains floats between 0 and 1
    predictions = model.predict(x_comb)

    for i in range(len(x_comb)):
        sequence = convert_to_simple_list(x_comb[i])
        pred_list = convert_to_simple_list(predictions[i])

        decoded_sequence = detokenize(sequence)
        create_better_heatmap(pred_list, decoded_sequence, i)


def convert_to_simple_list(complex_list):
    new_list = []
    for val in complex_list:
        new_list.append(val[0])
    return new_list


def detokenize(sequence):
    with open('./AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    reverse_word_map = dict(map(reversed, encoder.word_index.items()))
    new_seq = [reverse_word_map.get(i, "") for i in sequence]
    # crop after the first empty string
    return new_seq[:new_seq.index("")]


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
            axis = -1)

    return weighted_loss


def create_better_heatmap(data, sequence, index):
    """Input: predictions from the model
    Output: Heatmaps according to the predictions for the whole sequence entered"""
    print("creating heatmap for index " + str(index))

    data = np.array(data[:len(sequence)], dtype = np.float32)

    data_list, sequence_list = create_blocks(data, sequence)
    print(data_list, sequence_list)
    data_list = np.reshape(data_list, (data_list.shape[1], data_list.shape[0]))
    sequence_list = np.reshape(sequence_list, (sequence_list.shape[1], sequence_list.shape[0]))
    print("------------------------------------------------")
    print(data_list.shape)

    """change the path to a folder to save the pictures in"""
    filename = "./AI/pictures/heatmaps/" + str(index) + ".png"

    plt.figure(dpi = 1000)
    sb.heatmap(data_list, xticklabels = False, yticklabels = False, vmin = 0.2, vmax = 0.8, cmap = "rocket_r", annot=sequence_list, fmt="")
    plt.savefig(filename, dpi = 1000, bbox_inches = "tight")


def create_blocks(list1, list2):
    """creates blocks of max size 20 so that every heatmap has a max length of 20"""
    block_size = 20
    num_blocks1 = len(list1) // block_size
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


# def blubby():
#     recall, precision, f1, mcc, auc = evaluate_model(model, encoder, seq, mcc, epi)
#
#     results = []
#     results.append({
#             'PDB ID': pdb_id,
#             'Recall': recall,
#             'Precision': precision,
#             'F1-Score': f1,
#             'AUC': auc
#         })
#
#     # Ergebnisse in CSV speichern
#     results_df = pd.DataFrame(results)
#     results_df.to_csv('./data/evaluation_results_2.csv', index=False)
#     print("Evaluation abgeschlossen und in 'evaluation_results_2.csv' gespeichert.")


def evaluate_model(predictions, model, encoder, sequence, true_binary_epitope):
    # Da das Modell Wahrscheinlichkeiten ausgibt, runde auf 0 oder 1
    predicted_binary = np.where(predictions >= 0.5, 1, 0)
    # Berechne die Metriken
    print( "test: ", true_binary_epitope, predicted_binary)
    auc = masked_auc(true_binary_epitope, predictions)
    recall = masked_recall(true_binary_epitope, predicted_binary)
    precision = masked_precision(true_binary_epitope, predicted_binary)
    f1 = masked_f1_score(true_binary_epitope, predicted_binary)
    mcc = masked_mcc(y_true=true_binary_epitope, y_pred=predicted_binary)
    print(recall, precision, f1, mcc, auc)

    return recall, precision, f1, mcc ,auc


if __name__=="__main__":
    use_model_and_predict_ma()