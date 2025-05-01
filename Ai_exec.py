import numpy as np
import tf_keras
from tf_keras import optimizers as opt, layers
from transformers import  TFEsmForTokenClassification
import tensorflow as tf
import tensorflow
from tensorflow.keras import backend as K

from Master_Thesis_AI.utils.data_loading_generator import EpitopeDataGenerator
from ai_functionality_new import LayerGroup
from ai_functionality_old import embedding, modify_with_context, calculating_class_weights, \
    get_weighted_loss, get_weighted_loss_masked,  save_ai, use_model_and_predict, new_embedding, modify_with_context_big_dataset, \
    embedding_incl_structure

import logging

from src.masked_metrics import masked_accuracy, masked_recall, masked_precision
from validate_45_blind import validate_on_45_blind
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import mixed_precision

from validate_BP3C50ID_external_test_set import validate_on_BP3C59ID_external_test_set


def create_ai(filepath, save_file, output_file, train=False, safe=False, validate_45_Blind=False, validate_BP3C=False, predict=False, old=False, gpu_split=False, big_dataset=True, use_structure=False):

    if old==False:
        from ai_functionality_new import TokenAndPositionEmbedding_for_ESM, TransformerBlock, TransformerDecoderTwo
    else:
        from src.TransformerDecoderTwo import TransformerDecoderTwo
        from src.TokenAndPositionEmbedding import TokenAndPositionEmbedding
        from src.TransformerBlock import TransformerBlock

    if use_structure:
        embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder, structure_data = embedding_incl_structure(filepath, pdb_dir="./data/alphafold_structures_02", old=old)
        print(structure_data)
    else:
        embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder = embedding(filepath, old=old)



    # optimizersgd = opt.sgd_experimental.SGD(learning_rate=0.001, clipnorm=5)

    antigen_list = embedded_docs[:-300]
    epitope_list = epitope_embed_list[:-300]

    testx_list = embedded_docs[-300:]
    testy_list = epitope_embed_list[-300:]

    antigen_list_full_sequence = antigen_list
    epitope_list_full_sequence = epitope_list


    if big_dataset:
        epitope_list, antigen_list, length_of_longest_context = modify_with_context_big_dataset(epitope_list, antigen_list,  length_of_longest_sequence)
        testy_list, testx_list, length_of_longest_context_2 = modify_with_context_big_dataset(testy_list, testx_list,
                                                                                  length_of_longest_sequence)

    else:
        epitope_list, antigen_list, length_of_longest_context = modify_with_context(epitope_list, antigen_list,
                                                                                length_of_longest_sequence)
        testy_list, testx_list, length_of_longest_context_2 = modify_with_context(testy_list, testx_list,
                                                                              length_of_longest_sequence)


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


    ###Classweights
    new_weights = calculating_class_weights(epitope_list)


    # weights = class_weight.compute_sample_weight(class_weight='balanced', y=epitope_array)
    # print(pd.Series(test_sample_weights).unique())
    embedding_dim = 4
    # model = load_model('/my_test_model_02(1).h5', compile=False)

    np.seterr(all = None, divide = None, over = 'warn', under = None, invalid = None)

    num_transformer_blocks = 2 # used to be 2
    num_decoder_blocks = 1 # used to be 1
    embed_dim = 24  # Embedding size for each token used to be 24
    num_heads = 40  # Number of attention heads; used to be 40
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer; used to be 32
    maxlen = length_of_longest_context
    rate = 0.1
    training = True
    output_dimension = embedding_dim

    some_class_weight = {0: 1.,
                         1: 3.}

    testx_list = testx_list.astype(np.float16)
    testy_list = testy_list.astype(np.float16)
    # train_gen = EpitopeDataGenerator(training_data, epitope_list, epitope_list_for_weights, batch_size=50)
    # val_gen = EpitopeDataGenerator(testx_list, testy_list, testy_for_weights, batch_size=50, shuffle=False)
    tf.get_logger().setLevel(logging.ERROR)
    if train:
        K.clear_session()
        strategy = tf.distribute.MirroredStrategy()


        # Erstellen Sie Ihr Modell innerhalb der Strategie
        with strategy.scope():
            #optimizer = tf_keras.optimizers.Adam(learning_rate=0.0001)  # 0.001 for old_model # 0,0001 for New Model
            optimizer = tf_keras.optimizers.AdamW(learning_rate=0.0001)
            # with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
            callback = tf_keras.callbacks.EarlyStopping(
                monitor = 'val_loss',
                min_delta = 0,
                patience = 10,
                verbose = 0,
                mode = 'auto',
                baseline = None,
                restore_best_weights = True)

            encoder_inputs = layers.Input(shape = (length_of_longest_context,), name = 'encoder_inputs')


            if old:
                embedding_layer = TokenAndPositionEmbedding(maxlen, voc_size, embed_dim)

                encoder_embed_out = embedding_layer(encoder_inputs)
                mask = embedding_layer.compute_mask(encoder_inputs)
                x = encoder_embed_out
                output_dimension = x.shape[2]

            else:

                esm_model = TFEsmForTokenClassification.from_pretrained("facebook/esm2_t36_3B_UR50D")

                # Eingabe vorbereiten
                #encoder_inputs = layers.Input(shape=(length_of_longest_context,), name='encoder_inputs', dtype=tf.int32)


                # Nur die Embeddings extrahieren
                with tf.GradientTape() as tape:
                    if old == False:
                        if gpu_split:
                            esm_embeddings = split_esm_on_4GPUs(encoder_inputs, esm_model)
                            x = esm_embeddings
                            output_dimension = x.shape[-1]  # without mean reduction

                        else:
                            outputs = esm_model(encoder_inputs, output_hidden_states=True)
                            esm_embeddings = outputs.hidden_states[-1] #outputs.hidden_states[-1] war am Besten!
                            # Embedding-Schicht in das Modell einfügen
                            x = esm_embeddings

                            output_dimension = x.shape[2]  #without mean reduction LATEST

                #output_dimension = x.shape[0]



            for i in range(num_transformer_blocks):
                transformer_block = TransformerBlock(output_dimension, num_heads, ff_dim, rate)
                x = transformer_block(x, training = training, mask=mask)

            x = layers.Dropout(rate = rate)(x)
            encoder_outputs = layers.Dense(embed_dim, activation = "sigmoid")(x)

            decoder_outputs = TransformerDecoderTwo(embed_dim, ff_dim, num_heads)(encoder_outputs = encoder_outputs,
                                                                                  training = training, mask=mask)

            for i in range(num_decoder_blocks):
                transformer_decoder = TransformerDecoderTwo(embed_dim, ff_dim, num_heads)
                decoder_outputs = transformer_decoder(decoder_outputs, training = training, mask=mask)

            # decoder_outputs = layers.GlobalAveragePooling1D()(decoder_outputs)
            decoder_outputs = layers.Dropout(rate = rate)(decoder_outputs)
            decoder_outputs = layers.Dense(12, activation = "relu", name = 'Not_the_last_Sigmoid')(decoder_outputs)
            decoder_outputs_final = layers.TimeDistributed(layers.Dense(1, activation = "sigmoid", name = 'Final_Sigmoid'))(
                decoder_outputs, mask=mask)

            model = tf_keras.Model(inputs = encoder_inputs, outputs = decoder_outputs_final)

            model.compile(optimizer, loss = "binary_crossentropy", ### used to be get_weighted_loss(new_weights)
                          metrics=[masked_accuracy, masked_precision, masked_recall, tf_keras.metrics.AUC()]
                          #weighted_metrics = ['accuracy', tf_keras.metrics.AUC(), tf_keras.metrics.Precision(), tf_keras.metrics.Recall()]
                        )
            # model.compile(optimizer, loss="binary_crossentropy", weighted_metrics=['accuracy', tf.keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])
            print("training_data:", training_data[0]) # debug
            history = model.fit(x = training_data, y = epitope_list, batch_size = 50, epochs = 100,
                            validation_data = (testx_list, testy_list), callbacks = [callback], verbose=1)
        # history = model.fit(x=antigen_list, y=epitope_list, batch_size=50, epochs=100, validation_data=(testx_list, testy_list, testy_for_weights), callbacks=[callback], sample_weight = epitope_list_for_weights)

        # plot_results(history)

#

        tf_keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
                                  to_file = './multi_model' + str(i) + '.png')
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
        validate_on_BP3C59ID_external_test_set()
    amino_acid_counts_epitope_predicted, confusion_matrices = analyze_amino_acids_in_validation_data( model, validation_sequences=testx_list, validation_labels=testy_list, encoder=encoder)


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


