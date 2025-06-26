import tensorflow as tf
import tf_keras
import keras_hub
from sklearn.utils import compute_class_weight
from tf_keras import layers
from tf_keras.src.callbacks import ModelCheckpoint
from transformers import  TFEsmForTokenClassification
from tensorflow.keras import backend as K
import keras
import pickle
import pandas as pd
import ast
from tensorflow.keras.models import load_model
import json
import matplotlib.pyplot as plt
from datetime import datetime


from sklearn.model_selection import KFold

from Master_Thesis_AI.FusionModel import FusionModel
from ai_functionality_new import LayerGroup
from ai_functionality_old import embedding, modify_with_context, calculating_class_weights, \
    get_weighted_loss_masked, save_ai, use_model_and_predict, new_embedding, \
    modify_with_context_big_dataset, \
    embedding_incl_structure, get_weighted_loss_masked_, modify_with_max_epitope_density

import logging

from src.RemoveMask import RemoveMask
from src.TokenAndPositionEmbedding import TokenAndPositionEmbedding
from src.TransformerBlock import TransformerBlock
from src.TransformerDecoderTwo import TransformerDecoderTwo
from src.fusion_model import create_fusion_model_function, create_fusion_model_function_02
from src.masked_metrics import masked_accuracy, masked_recall, masked_precision, MaskedAUC, masked_f1_score, \
    masked_precision_metric, masked_recall_metric, masked_f1_score_metric
from validate_45_blind import validate_on_45_blind
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import mixed_precision

from validate_BP3C50ID_external_test_set import validate_on_BP3C59ID_external_test_set

from tensorflow.keras import mixed_precision



def load_structure_data(pickle_file):
    # Lade die Daten aus dem Pickle-File
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

        # Mappe die ID zu den Strukturdaten (z.B. Koordinaten und Sequenz), entferne "_alphafold"
        structure_map = {entry['id'].replace('_alphafold', ''): entry for entry in data}

    return structure_map


def pad_or_truncate(array, max_len=4562):
    if array.shape[0] > max_len:
        return array[:max_len]
    elif array.shape[0] < max_len:
        padding = np.zeros((max_len - array.shape[0], array.shape[1]), dtype=array.dtype)
        return np.vstack((array, padding))
    return array

def get_structure_from_accession_id(accession_ids=None, max_len=4562):
    pickle_file = "./data/alphafold_structures_conv2d.pkl"
    structure_map = load_structure_data(pickle_file)

    # Liste zum Speichern der Strukturdaten
    structures = []
    for accession_id in accession_ids:
        pdb_id = str(accession_id)  # Ersetze durch eine gültige ID

        if pdb_id in structure_map:
            # Zugriff auf die Struktur anhand der ID
            structure_data = structure_map[pdb_id]
            structure_array = structure_data['structure_array']  # Hier wird der 'structure_array' Key verwendet
            #print(f"Strukturdaten für ID {pdb_id} gefunden.")
            structure_array = pad_or_truncate(structure_array, max_len=max_len)
            structures.append(structure_array)  # Struktur hinzufügen
        elif isinstance(pdb_id, str) and pdb_id.lower() == "nan":
            # Erstelle eine leere Struktur als Platzhalter
            structure_array = np.zeros((20, 3), dtype=np.float16)  # Leeres NumPy-Array als Platzhalter

            # Variante 2: Mit newaxis
            #structure_array = structure_array[:, np.newaxis]
            print(f"⚠️ Leere Struktur für ID {pdb_id} als Platzhalter verwendet.")
            structure_array = pad_or_truncate(structure_array, max_len=max_len)

            structures.append(structure_array)
        else:
            structure_array = np.zeros((20, 3), dtype=np.float16)  # Leeres NumPy-Array als Platzhalter
            structure_array = pad_or_truncate(structure_array, max_len=max_len)


            # Variante 2: Mit newaxis
            # structure_array = structure_array[:, np.newaxis]
            structures.append(structure_array)
            print(f"Keine Strukturdaten für ID {pdb_id} gefunden.")
    return structures


def create_ai(filepath, save_file, output_file, train=False, safe=False, validate_45_Blind=False, validate_BP3C=False, predict=False, old=False, gpu_split=False, big_dataset=False, use_structure=False, ba_ai=False, full_length=False, old_data_set=False):
    #disable_eager_execution()
    mixed_precision.set_global_policy('mixed_float16')
    if old_data_set:
        if old==False:
            pass
        else:
            from src.TokenAndPositionEmbedding import TokenAndPositionEmbedding

        if use_structure:
            embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder, structure_data = embedding_incl_structure(filepath, pdb_dir="./data/alphafold_structures_02", old=old)
        else:
            embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder, accession_ids = embedding(filepath, old=old)



        antigen_list_accession_ids = accession_ids[:-300]
        antigen_list_structures = get_structure_from_accession_id(antigen_list_accession_ids, max_len=length_of_longest_sequence)
        antigen_list_structures = np.array(antigen_list_structures, dtype=np.float16)

        print(f"Strukturen geladen: {len(antigen_list_structures)}")

        antigen_list = embedded_docs[:-300]
        epitope_list = epitope_embed_list[:-300]
        print("Hier mal ein Test-Epitop: ", epitope_embed_list[0])
        print("Größe des Trainingsdatensatzes: ", len(antigen_list))

        testx_list_accession_ids = accession_ids[-300:]
        testx_list_structures = get_structure_from_accession_id(testx_list_accession_ids, max_len=length_of_longest_sequence )
        testx_list_structures = np.array(testx_list_structures, dtype=np.float16)


        testx_list = embedded_docs[-300:]
        testy_list = epitope_embed_list[-300:]
        print(f"Anzahl accession_ids insgesamt: {len(accession_ids)}")
        print(f"Anzahl für Trainingsdaten (ohne Testdaten): {len(antigen_list_accession_ids)}")

        #antigen_list_structures_padded = pad_sequences(antigen_list_structures, padding='post', value=0.0, dtype='float16', maxlen=4700)
        #testx_list_structures_padded = pad_sequences(testx_list_structures, padding='post', value=0.0, dtype='float16', maxlen=4700)

        #antigen_list = one_hot_embed[:-300] # test for one_hot_endcoding
        #testx_list = one_hot_embed[-300:] # test for one_hot_endcoding

        antigen_list_full_sequence = antigen_list
        epitope_list_full_sequence = epitope_list

    else:
        voc_size=100
        filepath = "./data/BP3_Data/BP3_training_set_transformed.csv"
        df_bp = pd.read_csv(filepath)

        #antigen_list = df_bp['Sequenz'].tolist()
        #epitope_list = df_bp['Epitop'].tolist()

        # String-Listen in echte Listen umwandeln
        antigen_list = [np.fromstring(seq_str.strip("[]"), sep=' ') for seq_str in df_bp['Sequenz']]
        epitope_list = [np.fromstring(seq_str.strip("[]"), sep=' ') for seq_str in df_bp['Epitop']]

        # In NumPy-Arrays konvertieren
        antigen_array = np.array(antigen_list, dtype=np.float16)
        epitope_array = np.array(epitope_list, dtype=np.float16)
        epitope_array.reshape(epitope_array.shape[0], epitope_array.shape[1], 1)




    try:
        if big_dataset:
            epitope_list, antigen_list, length_of_longest_context = modify_with_context_big_dataset(epitope_list, antigen_list,  length_of_longest_sequence)
            testy_list, testx_list, length_of_longest_context_2 = modify_with_context_big_dataset(testy_list, testx_list,
                                                                                      length_of_longest_sequence)

        elif full_length==False:
            epitope_list, antigen_list, length_of_longest_context = modify_with_context(epitope_list, antigen_list,
                                                                                    length_of_longest_sequence)
            testy_list, testx_list, length_of_longest_context_2 = modify_with_context(testy_list, testx_list,
                                                                                  length_of_longest_sequence)
        else:

            epitope_list, antigen_list, window_size = modify_with_max_epitope_density(epitope_list, antigen_list, window_size=length_of_longest_sequence)
            #epitope_list = epitope_list_full_sequence
            #antigen_list = antigen_list_full_sequence
            length_of_longest_context = length_of_longest_sequence


        if old==False:
            antigen_list = new_embedding(antigen_list, encoder)
            testx_list = new_embedding(testx_list, encoder)

        epitope_list_for_weights = epitope_list
        epitope_list_for_weights = np.array(epitope_list_for_weights, dtype = np.float16)

        epitope_list = np.array(epitope_list, dtype = np.float16)
        if old==True:
            antigen_list = np.array(antigen_list, dtype = np.float32)
            antigen_list = np.reshape(antigen_list, (antigen_list.shape[0], antigen_list.shape[1], 1))
            testx_list = np.array(testx_list, dtype = np.float32)
            testx_list = np.reshape(testx_list, (testx_list.shape[0], testx_list.shape[1], 1))

        if use_structure:
            training_data = get_training_data(antigen_list)
        else:
            training_data = antigen_list



        epitope_list_for_weights = np.reshape(epitope_list_for_weights,
                                              (epitope_list_for_weights.shape[0], epitope_list_for_weights.shape[1]))
        epitope_list = np.reshape(epitope_list, (epitope_list.shape[0], epitope_list.shape[1], 1))



        testy_list_for_weights = np.array(testy_list, dtype = np.float32)
        testy_list = np.array(testy_list, dtype = np.float32)

        testy_for_weights = np.reshape(testy_list_for_weights,
                                       (testy_list_for_weights.shape[0], testy_list_for_weights.shape[1]))
        testy_list = np.reshape(testy_list, (testy_list.shape[0], testy_list.shape[1], 1))



        for i, epitope in enumerate(testy_for_weights):
            for y, char in enumerate(epitope):
                if char == 0.:
                    testy_for_weights[i][y] = 0.1
                if char == 1.:
                    testy_for_weights[i][y] = 0.5

        for i, epitope in enumerate(epitope_list_for_weights):
            for y, char in enumerate(epitope):
                if char == 0.:
                    epitope_list_for_weights[i][y] = 0.1
                if char == 1.:
                    epitope_list_for_weights[i][y] = 0.5
        new_weights = calculating_class_weights(epitope_list)
    except:

        epitope_list = np.reshape(epitope_array, (epitope_array.shape[0],epitope_array.shape[1], 1))
        new_weights = calculating_class_weights(epitope_list)


    ###Classweights



    # weights = class_weight.compute_sample_weight(class_weight='balanced', y=epitope_array)
    # print(pd.Series(test_sample_weights).unique())
    embedding_dim = 4
    # model = load_model('/my_test_model_02(1).h5', compile=False)

    np.seterr(all = None, divide = None, over = 'warn', under = None, invalid = None)

    num_transformer_blocks = 2 # used to be 2
    num_decoder_blocks = 2 # used to be 1
    embed_dim = 40  # Embedding size for each token used to be 24
    num_heads = 40  # Number of attention heads; used to be 40
    ff_dim = 80  # Hidden layer size in feed forward network inside transformer; used to be 32
    try:
        maxlen = length_of_longest_context
    except:
        maxlen = antigen_array.shape[1]
        length_of_longest_context= maxlen

    rate = 0.1
    training = True
    output_dimension = embedding_dim

    try:
        testx_list = testx_list.astype(np.float16)
        testy_list = testy_list.astype(np.float16)
    except:
        pass
    # train_gen = EpitopeDataGenerator(training_data, epitope_list, epitope_list_for_weights, batch_size=50)
    # val_gen = EpitopeDataGenerator(testx_list, testy_list, testy_for_weights, batch_size=50, shuffle=False)
    tf.get_logger().setLevel(logging.ERROR)
    if train:
        K.clear_session()
        strategy = tf.distribute.MirroredStrategy()


        # Erstellen Sie Ihr Modell innerhalb der Strategie
        with strategy.scope():
            #optimizer = tf_keras.optimizers.Adam(learning_rate=0.0001) # 0.001 for old_model # 0,0001 for New Model
            # with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU

            early_stopping = tf_keras.callbacks.EarlyStopping(
                monitor = 'val_loss',
                patience = 20,
                min_delta=0.0001,
                verbose = 0,
                mode = 'auto',
                baseline = None,
                restore_best_weights = True)





            #i, model = create_model_old(embed_dim, ff_dim, gpu_split, i, length_of_longest_context, maxlen, new_weights,
                                        #num_decoder_blocks, num_heads, num_transformer_blocks, old,
                                        #output_dimension, rate, training, voc_size)
            # model.compile(optimizer, loss="binary_crossentropy", weighted_metrics=['accuracy', tf.keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])

            if ba_ai:
                results_per_fold = []
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                for fold, (train_index, test_index) in enumerate(kf.split(antigen_array)):
                    X_train, X_test = antigen_array[train_index], antigen_array[test_index]
                    y_train, y_test = epitope_array[train_index], epitope_array[test_index]
                    y_train = tf.expand_dims(y_train, axis=-1)
                    y_test = tf.expand_dims(y_test, axis=-1)
                    print(f"Fold {fold + 1}")
                    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
                    # Hier kannst du dann mit dem Training starten
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # to save the best model
                    checkpoint_filepath = f"./{timestamp}_best_model_fold_{fold + 1}.keras"
                    checkpoint_callback = ModelCheckpoint(
                        filepath=checkpoint_filepath,
                        monitor='val_loss',  # oder 'val_accuracy', je nach Metrik
                        save_best_only=True,
                        save_weights_only=False,
                        mode='min',  # 'min' für Verlust, 'max' für Accuracy
                        verbose=1
                    )

                    model = create_model_new(embed_dim, ff_dim, length_of_longest_context, maxlen, new_weights,
                                        num_decoder_blocks, num_heads, num_transformer_blocks, old, rate,
                                        voc_size)
                    history = model.fit(x = X_train,
                                        y = y_train,
                                        batch_size = 16,
                                        epochs = 100,
                                        validation_data = (X_test, y_test),
                                        callbacks = [early_stopping, checkpoint_callback],
                                        verbose=1)

                    history_dict = history.history
                    print(f"🔍 Fold {fold + 1} — History Keys:", list(history_dict.keys()))
                    print(history_dict)

                    plot_save_model_trianing_history(fold, history_dict, timestamp)
                    results_per_fold = load_and_evaluate_folds(X_test, X_train, checkpoint_filepath, fold, new_weights, results_per_fold,
                                            y_test, y_train)
                save_history_and_plot(results_per_fold, timestamp)



            elif use_structure:


                model = create_fusion_model_function(embed_dim, ff_dim, length_of_longest_context, maxlen, new_weights,
                                                     num_decoder_blocks, num_heads, num_transformer_blocks, old, rate,
                                                     voc_size)
                print("training_data.shape: ", training_data.shape)
                print("antigen_list.shape: ", antigen_list_structures.shape)
                print("Epitope_list.shape: ", epitope_list.shape)
                antigen_list_structures = tf.cast(antigen_list_structures, tf.float16)
                training_data = tf.cast(training_data, tf.float16)
                epitope_list = tf.cast(epitope_list, tf.float16)
                testx_list = tf.cast(testx_list, tf.float16)
                testx_list_structures = tf.cast(testx_list_structures, tf.float16)
                testy_list = tf.cast(testy_list, tf.float16)

                history = model.fit(x=[training_data, antigen_list_structures],
                                    y=epitope_list,
                                    batch_size=16,
                                    epochs=100,
                                    validation_data=([testx_list, testx_list_structures], testy_list),
                                    callbacks=[early_stopping],
                                    verbose=1)

            else:
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                results_per_fold = []

                for fold, (train_index, test_index) in enumerate(kf.split(antigen_array)):
                    X_train, X_test = antigen_array[train_index], antigen_array[test_index]
                    y_train, y_test = epitope_array[train_index], epitope_array[test_index]
                    y_train = tf.expand_dims(y_train, axis=-1)
                    y_test = tf.expand_dims(y_test, axis=-1)
                    print(f"Fold {fold + 1}")
                    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

                    # to save the best model
                    checkpoint_filepath = f"./best_model_fold_{fold + 1}.keras"
                    checkpoint_callback = ModelCheckpoint(
                        filepath=checkpoint_filepath,
                        monitor='val_loss',  # oder 'val_accuracy', je nach Metrik
                        save_best_only=True,
                        save_weights_only=False,
                        mode='min',  # 'min' für Verlust, 'max' für Accuracy
                        verbose=1
                    )

                    model = create_model_new(embed_dim, ff_dim, length_of_longest_context, maxlen, new_weights,
                                                num_decoder_blocks, num_heads, num_transformer_blocks, old, rate,
                                                voc_size)
                    history = model.fit(x=X_train,
                                        y=y_train,
                                        batch_size=40,
                                        epochs=100,
                                        validation_data=(X_test, y_test),
                                        callbacks=[early_stopping],
                                        verbose=1)

                    fold_result = load_and_evaluate_folds(X_test, X_train, checkpoint_filepath, fold, new_weights, results_per_fold,
                                            y_test, y_train)
                    results_per_fold.append(fold_result)
                save_history_and_plot(results_per_fold)

        # history = model.fit(x=antigen_list, y=epitope_list, batch_size=50, epochs=100, validation_data=(testx_list, testy_list, testy_for_weights), callbacks=[callback], sample_weight = epitope_list_for_weights)

        # plot_results(history)

#

        tf_keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
                                  to_file = './multi_model' + str("_test_") + '.png') #str("_test_") used to be str("i")
        if safe:
            save_ai(model, save_file, old=old)
        # load_model_and_do_stuff(testx_list, testy_list, model)
        # load_model_and_do_stuff(antigen_list, epitope_list, model)


        # transformer_evaluation_loop(test_trainx2, test_trainy2, test_trainy)
        # transformer_prediction_loop(trainx2, trainy2, trainy)
        # tf.saved_model.save(transformer, '/content/drive/MyDrive/ifp/model_test_saves.h5')

        # loss, accuracy, precision, recall = model.evaluate(testx, testy, verbose=1)

        # prediction = model.predict(x=trainx)

        # print(transformer.summary(expand_nested=True))

        # new_function([testx], testy, length_of_longest_context, length_of_longest_sequence, transformer)

        # print(len(antigen_list[0]))
        # print("Prediction for x: " + str(prediction))
        # print(len(prediction))
        # print(trainx[0])
        # print("trainy: " + str(trainy_2[1]))

        # plot_sth(history)

        # print(model.predict(test_x))
        # print(test_y)
        # print('Accuracy: %f' % (accuracy * 100))
        # print('Loss: %f' % (loss * 100))
        # print('Precision: %f' % (precision * 100))
        # print('Recall: %f' % (recall * 100))

        # print('testx_shape: ' + str(testx.shape))

        # load_model_and_do_stuff([trainx2[0], trainy2[0]], trainy, maxlen, voc_size, embed_dim, transformer, length_of_longest_context)

    if predict:
        use_model_and_predict()
    if validate_45_Blind:
        validate_on_45_blind()
    if validate_BP3C:
        validate_on_BP3C59ID_external_test_set(model=model)
    amino_acid_counts_epitope_predicted, confusion_matrices = analyze_amino_acids_in_validation_data( model, validation_sequences=testx_list, validation_labels=testy_list, encoder=encoder)


def save_history_and_plot(results_per_fold, timestamp):
    import csv

    # Header mit genau den gewünschten Metriken
    header = ["fold", "split", "auc", "f1", "precision", "recall"]

    with open(f"k_fold_model_metrics_{timestamp}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for result in results_per_fold:
            fold = result["fold"]
            for split in ["train", "test"]:
                metrics_dict = result[split]

                # Hole die Metrikwerte anhand der richtigen Namen
                row = [
                    fold,
                    split,
                    float(metrics_dict.get("masked_auc", 0.0)),
                    float(metrics_dict.get("masked_f1_score", 0.0)),
                    float(metrics_dict.get("masked_precision", 0.0)),
                    float(metrics_dict.get("masked_recall", 0.0)),
                ]
                writer.writerow(row)
    print("model saved in" + f"k_fold_model_metrics_{timestamp}.csv")


def load_and_evaluate_folds(X_test, X_train, checkpoint_filepath, fold, new_weights, results_per_fold, y_test, y_train):
    # Load best model and evaluate on both sets
    best_model = load_model(checkpoint_filepath,
                            compile=False,
                            safe_mode = False,
                            custom_objects={
                                "tf": tf
                                #,
                                #"MaskedAUC": MaskedAUC,
                                #"masked_precision": masked_precision_metric,
                                #"masked_recall": masked_recall_metric,
                                #"masked_f1_score": masked_f1_score_metric
                                #,"get_weighted_loss_masked_": get_weighted_loss_masked_(new_weights)
                            })
    # Modell nach dem Laden neu kompilieren
    best_model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001),
        loss=get_weighted_loss_masked_(new_weights),
        metrics=[
            MaskedAUC(name="masked_auc"),
            tf.keras.metrics.MeanMetricWrapper(masked_precision, name="masked_precision"),
            tf.keras.metrics.MeanMetricWrapper(masked_recall, name="masked_recall"),
            tf.keras.metrics.MeanMetricWrapper(masked_f1_score, name="masked_f1_score")
        ]
    )

    train_metrics = best_model.evaluate(X_train, y_train, verbose=2, return_dict=True)
    test_metrics = best_model.evaluate(X_test, y_test, verbose=2, return_dict=True)

    print(train_metrics, test_metrics)
    # Collect the metric names
    metric_names = ["loss", "masked_auc", "masked_recall", "masked_precision", "masked_f1_score"]
    results_per_fold.append({
        "fold": fold + 1,
        "train": train_metrics,
        "test": test_metrics
    })
    return results_per_fold


def plot_save_model_trianing_history(fold, history_dict, timestamp):
    #to save
    with open(f"./training_histories/_{timestamp}_history_fold_{fold + 1}.json", "w") as f:
        json.dump(history_dict, f)
    # Plot each metric
    for metric in history_dict:
        if metric.startswith("val_"):
            continue  # We'll plot val_* in same figure

        val_metric = "val_" + metric
        plt.figure()
        plt.plot(history_dict[metric], label=f"Train {metric}")
        if val_metric in history_dict:
            plt.plot(history_dict[val_metric], label=f"Val {metric}")
        plt.title(f"Fold {fold + 1} - {metric}")
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./training_plots/_{timestamp}_fold_{fold + 1}_{metric}.png")
        plt.close()


def create_fusionmodel(embed_dim, ff_dim, length_of_longest_context, maxlen, new_weights, num_decoder_blocks,
                     num_heads, num_transformer_blocks, old, rate, voc_size):

    structure_input_length: int = length_of_longest_context
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)
    fusion_model = FusionModel(
        length_of_longest_context=length_of_longest_context,
        voc_size=voc_size,
        embed_dim=embed_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        num_transformer_encoder_blocks=num_transformer_blocks,
        num_decoder_blocks=num_decoder_blocks,
        rate=rate
    )

    # build model
    fusion_model.build(input_shape=[(None, maxlen), (None, structure_input_length)])


    # compile model
    fusion_model.compile(
        optimizer=optimizer,
        loss=get_weighted_loss_masked(new_weights),
        metrics=[MaskedAUC(),
            masked_precision,
            masked_recall,
            masked_f1_score]
    )
    return fusion_model


def create_model_new(embed_dim, ff_dim, length_of_longest_context, maxlen, new_weights, num_decoder_blocks,
                     num_heads, num_transformer_blocks, old, rate, voc_size):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001)

    encoder_inputs = keras.layers.Input(shape=(length_of_longest_context,), name='encoder_inputs')
    # Instanziiere das Layer mit den Gewichtungen
    if old:
        embedding_layer = keras_hub.layers.TokenAndPositionEmbedding(voc_size, maxlen, embed_dim, mask_zero=True)
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
        )(x)
    encoder_outputs = keras.layers.Dense(embed_dim, activation='sigmoid')(x)
    # Decoder
    decoder_outputs = encoder_outputs
    for i in range(num_decoder_blocks):
        decoder_outputs = keras_hub.layers.TransformerDecoder(
            intermediate_dim=output_dimension,
            num_heads=num_heads,
            dropout=rate
        )(decoder_outputs, encoder_outputs)
    decoder_outputs = keras.layers.Dropout(rate)(decoder_outputs)
    decoder_outputs = keras.layers.Dense(24, activation='relu', name='Not_the_last_Sigmoid')(decoder_outputs)
    #decoder_outputs = keras.layers.Dropout(rate)(decoder_outputs)

    #decoder_outputs = keras.layers.Dense(8, activation='relu', name='Not_the_last_Sigmoid_02')(decoder_outputs)
    #decoder_outputs = keras.layers.Dropout(rate)(decoder_outputs)

    #decoder_outputs = keras.layers.Dense(8, activation='relu', name='Not_the_last_Sigmoid_03')(decoder_outputs)
    #decoder_outputs = keras.layers.Dropout(rate)(decoder_outputs)

    #decoder_outputs = keras.layers.Dense(4, activation='relu', name='Not_the_last_Sigmoid_04')(decoder_outputs)
    """"
    decoder_outputs = keras.layers.Lambda(lambda x: tf.identity(x),
    output_shape=lambda s: s )(decoder_outputs) # removes mask for timedistributed layer since it cant deal with a mask
    """
    decoder_outputs = RemoveMask()(decoder_outputs)
    decoder_outputs_final = keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid', name='Final_Sigmoid'))(
        decoder_outputs)
    model = keras.Model(inputs=encoder_inputs, outputs=decoder_outputs_final)
    model.compile(
        optimizer=optimizer,
        loss=get_weighted_loss_masked_(new_weights),
        metrics=[#masked_accuracy,
            MaskedAUC(),
            masked_precision_metric,
            masked_recall_metric,
            masked_f1_score_metric]
    )
    return model




def create_model_old(embed_dim, ff_dim, gpu_split, i, length_of_longest_context, maxlen, new_weights,
                     num_decoder_blocks, num_heads, num_transformer_blocks, old, output_dimension, rate,
                     training, voc_size):
    optimizer = tf_keras.optimizers.AdamW(learning_rate=0.0001)

    encoder_inputs = layers.Input(shape=(length_of_longest_context,), name='encoder_inputs')
    if old:
        embedding_layer = TokenAndPositionEmbedding(maxlen, voc_size, embed_dim)

        encoder_embed_out = embedding_layer(encoder_inputs)
        mask = embedding_layer.compute_mask(encoder_inputs)
        x = encoder_embed_out
        output_dimension = x.shape[2]

    else:

        esm_model = TFEsmForTokenClassification.from_pretrained("facebook/esm2_t36_3B_UR50D")

        # Eingabe vorbereiten
        # encoder_inputs = layers.Input(shape=(length_of_longest_context,), name='encoder_inputs', dtype=tf.int32)

        # Nur die Embeddings extrahieren
        with tf.GradientTape() as tape:
            if old == False:
                if gpu_split:
                    esm_embeddings = split_esm_on_4GPUs(encoder_inputs, esm_model)
                    x = esm_embeddings
                    output_dimension = x.shape[-1]  # without mean reduction

                else:
                    outputs = esm_model(encoder_inputs, output_hidden_states=True)
                    esm_embeddings = outputs.hidden_states[-1]  # outputs.hidden_states[-1] war am Besten!
                    # Embedding-Schicht in das Modell einfügen
                    x = esm_embeddings

                    output_dimension = x.shape[2]  # without mean reduction LATEST

        # output_dimension = x.shape[0]
    for i in range(num_transformer_blocks):
        transformer_block = TransformerBlock(output_dimension, num_heads, ff_dim, rate)
        x = transformer_block(x, training=training, mask=mask)
    x = layers.Dropout(rate=rate)(x)
    encoder_outputs = layers.Dense(embed_dim, activation="sigmoid")(x)
    decoder_outputs = TransformerDecoderTwo(embed_dim, ff_dim, num_heads)(encoder_outputs=encoder_outputs,
                                                                          training=training, mask=mask)
    for i in range(num_decoder_blocks):
        transformer_decoder = TransformerDecoderTwo(embed_dim, ff_dim, num_heads)
        decoder_outputs = transformer_decoder(decoder_outputs, training=training, mask=mask)
    # decoder_outputs = layers.GlobalAveragePooling1D()(decoder_outputs)
    decoder_outputs = layers.Dropout(rate=rate)(decoder_outputs)
    decoder_outputs = layers.Dense(12, activation="relu", name='Not_the_last_Sigmoid')(decoder_outputs)
    decoder_outputs_final = layers.TimeDistributed(layers.Dense(1, activation="sigmoid", name='Final_Sigmoid'))(
        decoder_outputs, mask=mask)
    model = tf_keras.Model(inputs=encoder_inputs, outputs=decoder_outputs_final)
    model.compile(optimizer, loss=get_weighted_loss_masked(new_weights),  ### used to be get_weighted_loss(new_weights)
                  metrics=[masked_accuracy, masked_precision, masked_recall, tf_keras.metrics.AUC()]
                  # weighted_metrics = ['accuracy', tf_keras.metrics.AUC(), tf_keras.metrics.Precision(), tf_keras.metrics.Recall()]
                  )
    return i, model


def get_training_data(antigen_list, structure_data):
    """returns a list of training data for the model where antigen_list contains the sequences and structure_data the corresponding structures"""
    return [antigen_list, structure_data]


def split_esm_on_4GPUs(encoder_inputs, esm_model):
    # outputs = esm_model(encoder_inputs, output_hidden_states=True)
    # Aufteilen der Transformer-Layer und sie zu Modellen umwandeln
    # Extrahiere die Schichten des Modells und teile sie auf
    all_layers = esm_model.layers
    # Anzahl der Layer teilen und auf GPUs verteilen
    num_layers = len(all_layers)
    split_size = num_layers // 4
    x = encoder_inputs
    # Layer-Gruppen erstellen
    with tf.device('/GPU:0'):
        part1_layers = all_layers[:split_size]
        part1_model = LayerGroup(part1_layers)
        part1_outputs = part1_model.call(x, training=False)
    with tf.device('/GPU:1'):
        part2_layers = all_layers[split_size:2 * split_size]
        part2_model = LayerGroup(part2_layers)
        part2_outputs = part2_model.call(part1_outputs, training=False)
    with tf.device('/GPU:2'):
        part3_layers = all_layers[2 * split_size:3 * split_size]
        part3_model = LayerGroup(part3_layers)
        part3_outputs = part3_model.call(part2_outputs, training=False)
    with tf.device('/GPU:3'):
        part4_layers = all_layers[3 * split_size:-2]
        part4_model = LayerGroup(part4_layers)
        outputs = part4_model.call(part3_outputs, training=False)
    esm_embeddings = outputs[0]
    print("These are the outputs", outputs)
    return esm_embeddings


import numpy as np
from sklearn.metrics import confusion_matrix


def analyze_amino_acids_in_validation_data(
    model,
    validation_sequences,
    validation_labels,
    encoder,
    batch_size=4
):
    """
    1) Dekodiert die Validierungssequenzen über den Encoder.
       (sequences_to_texts)
    2) Entfernt die Leerzeichen, um reine Aminosäurestrings zu erhalten.
    3) Führt die Vorhersage in Batches durch und konvertiert sie in 0/1.
    4) Erstellt pro Aminosäure eine Confusionsmatrix über alle Sequenzen.

    Parameter
    ---------
    model : Beliebiges Modell mit predict()-Methode
        Ihr trainiertes Modell, das Epitope (0/1) vorhersagen kann.
    validation_sequences : list oder np.array
        Liste/Array aller Sequenzen (Aminosäure-Indices),
        Shape: (n, seq_len)
    validation_labels : list oder np.array
        Liste/Array der wahren Labels (0/1) in gleicher Reihenfolge/Größe
        wie validation_sequences, Shape: (n, seq_len)
    encoder :
        Ein Objekt (z.B. Keras Tokenizer) mit einer Methode sequences_to_texts.
    batch_size : int
        Größe der Batches für die Vorhersage.

    Returns
    -------
    dict
        Dictionary, das jeder Aminosäure (als Schlüssel) die entsprechende
        Confusionsmatrix (2x2) zuordnet.
    """

    # 1) Dekodierung aller Sequenzen zu Text
    decoded_antigens: list = encoder.sequences_to_texts(validation_sequences)

    # 2) Leerzeichen entfernen
    for i, decoded_antigen in enumerate(decoded_antigens):
        decoded_antigens[i] = decoded_antigen.replace(" ", "")

    # 3) Batches für Vorhersagen
    all_predictions = []
    for start_idx in range(0, len(validation_sequences), batch_size):
        batch_seq = validation_sequences[start_idx:start_idx+batch_size]
        batch_pred = model.predict(batch_seq)

        # Konvertieren zu 0 und 1 (Schwellwert 0.5)
        batch_pred = (batch_pred > 0.5).astype(int)
        all_predictions.append(batch_pred)

    predictions = np.concatenate(all_predictions, axis=0)

    # 4) Pro Aminosäure Confusionsmatrix erstellen
    amino_acid_true_labels = {}
    amino_acid_pred_labels = {}

    for seq_idx, sequence_str in enumerate(decoded_antigens):
        true_seq = validation_labels[seq_idx]
        pred_seq = predictions[seq_idx]

        # Jede Position = ein Aminosäuren-Charakter
        for pos_idx, aa_symbol in enumerate(sequence_str):
            true_label = int(true_seq[pos_idx])
            pred_label = int(pred_seq[pos_idx])

            if aa_symbol not in amino_acid_true_labels:
                amino_acid_true_labels[aa_symbol] = []
                amino_acid_pred_labels[aa_symbol] = []

            amino_acid_true_labels[aa_symbol].append(true_label)
            amino_acid_pred_labels[aa_symbol].append(pred_label)

    confusion_matrices = {}
    for aa_symbol in amino_acid_true_labels:
        y_true = amino_acid_true_labels[aa_symbol]
        y_pred = amino_acid_pred_labels[aa_symbol]

        # Confusionsmatrix (labels=[0,1]):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        confusion_matrices[aa_symbol] = cm

        # Beispielhafte Ausgabe
        print(f"Aminosäure: {aa_symbol}")
        print("Confusionsmatrix [TN, FP; FN, TP]:")
        print(cm)
        print("-" * 30)

    return confusion_matrices


