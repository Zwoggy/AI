import argparse
import pickle
import sys

import keras
import numpy as np
import pandas as pd
import seaborn as sb
import tensorflow as tf
from keras_hub.layers import TransformerEncoder, TokenAndPositionEmbedding, TransformerDecoder
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K

from Master_Thesis_AI.src.validate_on_29_external import return_29_external_dataset_X_y
from src import RemoveMask
from src.masked_metrics import masked_mcc, masked_f1_score, masked_precision, masked_recall


def use_model_and_predict_ma(threshold, test_run = False):
    """Enter a sequence to use for prediction and generate the heatmap output.
    All path need to be changed to wherever the files are stored on your computer."""
    tf.keras.backend.clear_session()
    #TODO change the following path to the final_AI folder path
    model = keras.saving.load_model('./Master_Thesis_AI/models/20250928_121001_best_model_fold_no_k_fold.keras',
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
                              to_file = './Master_Thesis_AI/output/model_architecture.png', show_layer_activations = True)
    print(model.summary(expand_nested = True))

    x_comb, padded_epitope_list, id_list = return_29_external_dataset_X_y(model, maxlen=933, use_structure = True)
    # x_comb: array consisting of:
    # 29 proteins
    # 933 sequence (length of 933)
    # 8 infos (relevant: here only index 0)

    # contains floats between 0 and 1
    predictions = model.predict(x_comb)
    results = []

    # save results for every sequence
    for i in range(len(x_comb)):
        # create lists of [933] elements
        sequence = convert_to_simple_list(x_comb[i])
        pred_list = convert_to_simple_list(predictions[i])
        pdb_id = id_list[i]

        # collect data for csv file
        results.append(collect_evaluation_data(np.array(pred_list), padded_epitope_list[i], pdb_id, threshold))

        # create heatmap
        decoded_sequence = detokenize(sequence)
        pred_list = np.array(pred_list[:len(decoded_sequence)], dtype=np.float32)
        create_better_heatmap(pred_list, decoded_sequence, pdb_id)

        # create line plot
        create_line_plot(pred_list, pdb_id)

        # create csv file containing heatmap values for PyMOL
        create_pymol_heatmap_csv(pred_list, pdb_id)

        if test_run and i == 0:
            break

    # save csv file
    save_evaluation_result(results, threshold)


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


def create_better_heatmap(data, sequence, pdb_id):
    """Input: predictions from the model
    Output: Heatmaps according to the predictions for the whole sequence entered"""
    print("creating heatmap for pdb_id " + str(pdb_id))

    data_list, sequence_list = create_blocks(data, sequence)
    # print(data_list, sequence_list)
    data_list = np.reshape(data_list, (data_list.shape[1], data_list.shape[0]))
    sequence_list = np.reshape(sequence_list, (sequence_list.shape[1], sequence_list.shape[0]))
    # print(data_list.shape)

    filename = "./Master_Thesis_AI/output/heatmaps/" + str(pdb_id) + ".png"
    plt.figure(dpi = 1000)
    sb.heatmap(data_list, xticklabels = False, yticklabels = False, vmin = 0.2, vmax = 0.8, cmap = "rocket_r", annot=sequence_list, fmt="")
    plt.savefig(filename, dpi = 1000, bbox_inches = "tight")
    print("saved heatmap: " + filename)


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


def create_line_plot(data, pdb_id):
    filename = "./Master_Thesis_AI/output/plots/" + str(pdb_id) + ".png"

    plt.figure(dpi=1000)
    plt.style.use('_mpl-gallery')

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(data)), data, linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Index")
    ax.set_ylabel("Prediction")

    plt.savefig(filename, dpi=1000, bbox_inches="tight")
    print("saved plot: " + filename)


def collect_evaluation_data(predictions, true_epitope, pdb_id, threshold):
    recall, precision, f1, mcc = evaluate_model(predictions, true_epitope, threshold)

    return {
        'PDB ID': pdb_id,
        'Recall': tf.keras.backend.get_value(recall),
        'Precision': tf.keras.backend.get_value(precision),
        'F1-Score': tf.keras.backend.get_value(f1),
        'MCC': tf.keras.backend.get_value(mcc)
    }


def save_evaluation_result(results, threshold):
    # Ergebnisse in CSV speichern
    results_df = pd.DataFrame(results)
    threshold_str = str(threshold).replace(".", "_")
    results_df.to_csv('./Master_Thesis_AI/output/evaluation_results_' + threshold_str + '.csv', index=False)
    print("Evaluation abgeschlossen und in 'evaluation_results.csv' gespeichert.")


def evaluate_model(predictions, true_binary_epitope, threshold):
    # Da das Modell Wahrscheinlichkeiten ausgibt, runde auf 0 oder 1
    predicted_binary = np.where(predictions >= threshold, 1, 0)
    # Berechne die Metriken
    print( "test: ", true_binary_epitope, predicted_binary)
    #auc = masked_auc(true_binary_epitope, predictions) # TODO wenn Zeit, dann komische squeeze errors beheben
    recall = masked_recall(true_binary_epitope, predicted_binary)
    precision = masked_precision(true_binary_epitope, predicted_binary)
    f1 = masked_f1_score(true_binary_epitope, predicted_binary)
    mcc = masked_mcc(y_true=true_binary_epitope, y_pred=predicted_binary)
    print(recall, precision, f1, mcc)

    return recall, precision, f1, mcc


def create_pymol_heatmap_csv(data, pdb_id):
    start_index = get_startingindex_by_pdbid(pdb_id)

    csv_data = []
    for i, item in enumerate(data):
        csv_data.append({
            'index': i + start_index,
            'prediction': item
        })

    filename = str(pdb_id) + ".csv"
    results_df = pd.DataFrame(csv_data)
    results_df.to_csv('./Master_Thesis_AI/output/pymol/' + filename, index=False)
    print("saved " + filename)


def get_startingindex_by_pdbid(pdb_id):
    df = pd.read_csv('./data/Caroll_et_al_data/biomolecules_incl_sequences_and_epitopes.csv')
    try:
        return df.loc[df['PDB ID'] == pdb_id, 'Starting_Index'].iloc[0]
    except IndexError:
        return 0


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, required=False, default=0.5, help='Prediction Threshold (Default: 0.5)')
    parser.add_argument('--test', type=bool, required=False, default=False, help='Runs this script for the first sequence only.')
    args = parser.parse_args()

    print("run use_model_and_predict_ma with threshold=" + str(args.threshold) + ", test=" + str(args.test))

    use_model_and_predict_ma(args.threshold, args.test)
