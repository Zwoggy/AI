import numpy as np
import tf_keras
from tf_keras import optimizers as opt, layers
from transformers import  TFEsmForTokenClassification
import tensorflow as tf
import tensorflow
from tensorflow.keras import backend as K

from ai_functionality_new import LayerGroup
from ai_functionality_old import embedding, modify_with_context, calculating_class_weights, \
    get_weighted_loss, save_ai, use_model_and_predict, new_embedding, modify_with_context_big_dataset, \
    embedding_incl_structure

import logging

from validate_45_blind import validate_on_45_blind
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras import mixed_precision



def create_ai(filepath, save_file, output_file, train=False, safe=False,  validate=False, predict=False, old=False, gpu_split=False, big_dataset=True, use_structure=False):

    if old==False:
        from ai_functionality_new import TokenAndPositionEmbedding_for_ESM, TransformerBlock, TransformerDecoderTwo
    else: from ai_functionality_old import TransformerBlock, TransformerDecoderTwo, TokenAndPositionEmbedding

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
    epitope_list_for_weights = np.array(epitope_list_for_weights, dtype = np.float32)

    epitope_list = np.array(epitope_list, dtype = np.float32)
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



    """
    test_trainx2 = np.load("C:/Users/fkori/PycharmProjects/AI/data/test_data/test_antigen_array.npy")
    test_trainy2 = np.load("C:/Users/fkori/PycharmProjects/AI/data/test_data/test_epitope_array.npy")
    test_trainy = np.load("C:/Users/fkori/PycharmProjects/AI/data/test_data/test_epitope_array_train.npy")

    np.save("/epitope_array2.npy", epitope_array)
    np.save("/epitope_array_train2.npy", epitope_array_train)
    np.save("/antigen_array2.npy", antigen_array)

    epitope_array = np.load("/epitope_array2.npy")
    epitope_array_train = np.load("/epitope_array_train2.npy")
    antigen_array = np.load("/antigen_array2.npy")

    print("trainy")
    print(epitope_array_train.shape)
    print(antigen_array[0][0])

    #enable for evaluation
    trainx2 = np.reshape(antigen_array, ((antigen_array.shape[0] * antigen_array.shape[2]), antigen_array.shape[3]))
    trainy2 = np.reshape(epitope_array, ((epitope_array.shape[0] * epitope_array.shape[2]), epitope_array.shape[3]))
    trainy = np.reshape(epitope_array_train, ((epitope_array_train.shape[0] * epitope_array_train.shape[2]), 1))

    repeat_counter = []
    delete_counter = []
    for i in range(trainy.shape[0]):
        if trainy[i] == 2:
            delete_counter.append(i)


    trainx2 = np.delete(trainx2, delete_counter, axis = 0)
    trainy2 = np.delete(trainy2, delete_counter, axis = 0)
    trainy = np.delete(trainy, delete_counter, axis = 0)

    for i in range(trainy.shape[0]):
        if trainy[i] == 1:
            repeat_counter.append(4)
            ###war eben 5
        else:
            repeat_counter.append(1)

    trainx2 = np.repeat(trainx2, repeats = repeat_counter, axis = 0)
    trainy2 = np.repeat(trainy2, repeats = repeat_counter, axis = 0)
    trainy = np.repeat(trainy, repeats = repeat_counter, axis = 0)
    
    repeat_counter_for_validation_data = []
    delete_counter_for_validation_data = []
    for i in range(test_trainy.shape[0]):

        if test_trainy[i] == 2:
            delete_counter_for_validation_data.append(i)

    test_trainx2 = np.delete(test_trainx2, delete_counter_for_validation_data, axis = 0)
    test_trainy2 = np.delete(test_trainy2, delete_counter_for_validation_data, axis = 0)
    test_trainy = np.delete(test_trainy, delete_counter_for_validation_data, axis = 0)

    for i in range(test_trainy.shape[0]):
        if test_trainy[i] == 1:
            repeat_counter_for_validation_data.append(4)
        else:
            repeat_counter_for_validation_data.append(1)

    test_trainx2 = np.repeat(test_trainx2, repeats = repeat_counter_for_validation_data, axis = 0)
    test_trainy2 = np.repeat(test_trainy2, repeats = repeat_counter_for_validation_data, axis = 0)
    test_trainy = np.repeat(test_trainy, repeats = repeat_counter_for_validation_data, axis = 0)
    

    unique, counts = np.unique(trainy, return_counts = True)
    print(unique, counts)
    print(np.asarray((unique, counts)).T)
    unique, counts = np.unique(test_trainy, return_counts = True)
    print(unique, counts)
    print(np.asarray((unique, counts)).T)
    """
    ###Classweights
    new_weights = calculating_class_weights(epitope_list)


    # weights = class_weight.compute_sample_weight(class_weight='balanced', y=epitope_array)
    # print(pd.Series(test_sample_weights).unique())
    embedding_dim = 4
    # model = load_model('/my_test_model_02(1).h5', compile=False)

    np.seterr(all = None, divide = None, over = 'warn', under = None, invalid = None)

    num_transformer_blocks = 2
    num_decoder_blocks = 1
    embed_dim = 24  # Embedding size for each token used to be 24
    num_heads = 40  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    maxlen = length_of_longest_context
    rate = 0.1
    training = True
    output_dimension = embedding_dim

    some_class_weight = {0: 1.,
                         1: 3.}

    testx_list = testx_list.astype(np.float16)
    testy_list = testy_list.astype(np.float16)

    #set_global_policy('mixed_float16')
    #policy = mixed_precision.Policy('mixed_float16')
    #mixed_precision.set_global_policy(policy)
    tf.get_logger().setLevel(logging.ERROR)
    if train:
        K.clear_session()
        strategy = tf.distribute.MirroredStrategy()


        # Erstellen Sie Ihr Modell innerhalb der Strategie
        with strategy.scope():
            optimizer = tf_keras.optimizers.Adam(learning_rate=0.0001)  # 0.001 for old_model # 0,0001 for New Model
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
                x = transformer_block(x, training = training)

            x = layers.Dropout(rate = rate)(x)
            encoder_outputs = layers.Dense(embed_dim, activation = "sigmoid")(x)

            decoder_outputs = TransformerDecoderTwo(embed_dim, ff_dim, num_heads)(encoder_outputs = encoder_outputs,
                                                                                  training = training)

            for i in range(num_decoder_blocks):
                transformer_decoder = TransformerDecoderTwo(embed_dim, ff_dim, num_heads)
                decoder_outputs = transformer_decoder(decoder_outputs, training = training)

            # decoder_outputs = layers.GlobalAveragePooling1D()(decoder_outputs)
            decoder_outputs = layers.Dropout(rate = rate)(decoder_outputs)
            decoder_outputs = layers.Dense(12, activation = "sigmoid", name = 'Not_the_last_Sigmoid')(decoder_outputs)
            decoder_outputs_final = layers.TimeDistributed(layers.Dense(1, activation = "sigmoid", name = 'Final_Sigmoid'))(
                decoder_outputs)

            model = tf_keras.Model(inputs = encoder_inputs, outputs = decoder_outputs_final)

            model.compile(optimizer, loss = get_weighted_loss(new_weights),
                          weighted_metrics = ['accuracy', tf_keras.metrics.AUC(), tf_keras.metrics.Precision(),
                                              tf_keras.metrics.Recall()])
            # model.compile(optimizer, loss="binary_crossentropy", weighted_metrics=['accuracy', tf.keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])
            print("training_data:", training_data[0]) # debug
            history = model.fit(x = training_data, y = epitope_list, batch_size = 200, epochs = 100,
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
    if validate:
        validate_on_45_blind()
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


def analyze_amino_acids_in_validation_data(model, validation_sequences, validation_labels, encoder):
    """
    Bestimmt für alle Aminosäuren in den Validierungsdaten,
    wie oft sie als Epitope gekennzeichnet sind bzw. vorhergesagt werden,
    und erstellt Confusionsmatrizen für jede Aminosäure.

    Parameter
    ---------
    model : <Modell-Klasse>
        Ihr trainiertes Modell, das eine Vorhersage erzeugen kann.
    validation_sequences : list oder np.array
        Liste/Batch aller Sequenzen in den Validierungsdaten (Aminosäure-Indices).
    validation_labels : list oder np.array
        Passende Epitope-Labels (0/1) zu den Validierungsdaten,
        im selben Format/Shape wie validation_sequences.
    encoder : <Encoder-Klasse>
        Encoder (oder Mapping), der Indizes zu Aminosäure-Symbolen umwandelt.
    """

    # Vorhersagen vom Modell holen (Shape sollte dem der validation_labels entsprechen)
    predictions = model.predict(validation_sequences)  # Form: (batch_size, seq_length, 1) o.Ä.
    predictions = (predictions > 0.5).astype(int)  # In 0/1 konvertieren

    # Statistik-Strukturen vorbereiten
    amino_acid_counts_total = {}
    amino_acid_counts_epitope_true = {}
    amino_acid_counts_epitope_predicted = {}

    # Zum Speichern der Confusionsmatrix pro Aminosäure
    # dict: Aminosäure -> [TP, FP, FN, TN] oder als Sklearn-Matrix
    confusion_matrices = {}

    # Durch alle Sequenzen und Positionen iterieren
    for seq_idx, sequence in enumerate(validation_sequences):
        true_labels = validation_labels[seq_idx]
        pred_labels = predictions[seq_idx]

        for pos_idx, aa_index in enumerate(sequence):
            aa_symbol = encoder[aa_index]  # z.B. 'A', 'R', 'N' usw.
            true_label = int(true_labels[pos_idx])
            pred_label = int(pred_labels[pos_idx])

            # Gesamtzähler der Aminosäure aktualisieren
            amino_acid_counts_total[aa_symbol] = amino_acid_counts_total.get(aa_symbol, 0) + 1

            # Zähler, falls in Ground Truth ein Epitope
            if true_label == 1:
                amino_acid_counts_epitope_true[aa_symbol] = amino_acid_counts_epitope_true.get(aa_symbol, 0) + 1

            # Zähler, falls in Prediction ein Epitope
            if pred_label == 1:
                amino_acid_counts_epitope_predicted[aa_symbol] = amino_acid_counts_epitope_predicted.get(aa_symbol,
                                                                                                         0) + 1

    # Für die Confusionsmatrix werden alle Positionen derselben Aminosäure
    # gesammelt und einmalig gegenübergestellt
    # (Alternativ: man könnte Position für Position als Einzelbeispiel werten)

    # Alle gefundenen Aminosäuren ermitteln
    all_amino_acids = list(amino_acid_counts_total.keys())

    for aa_symbol in all_amino_acids:
        # Listen für Ground-Truth-Labels und Predictions, bezogen nur auf diese Aminosäure
        aa_true_labels = []
        aa_pred_labels = []
        for seq_idx, sequence in enumerate(validation_sequences):
            true_labels = validation_labels[seq_idx]
            pred_labels = predictions[seq_idx]
            for pos_idx, aa_index in enumerate(sequence):
                if encoder[aa_index] == aa_symbol:
                    aa_true_labels.append(int(true_labels[pos_idx]))
                    aa_pred_labels.append(int(pred_labels[pos_idx]))

        # Confusionsmatrix erstellen
        # confusion_matrix liefert eine 2x2-Matrix in der Reihenfolge [ [TN, FP], [FN, TP] ]
        cm = confusion_matrix(aa_true_labels, aa_pred_labels, labels=[0, 1])
        confusion_matrices[aa_symbol] = cm

    # Ausgabe oder Rückgabe der Ergebnisse
    for aa_symbol in all_amino_acids:
        total = amino_acid_counts_total[aa_symbol]
        true_epi = amino_acid_counts_epitope_true.get(aa_symbol, 0)
        pred_epi = amino_acid_counts_epitope_predicted.get(aa_symbol, 0)
        cm = confusion_matrices[aa_symbol]

        print(f"Aminosäure: {aa_symbol}")
        print(f"- Gesamt vorkommend: {total}")
        print(f"- Tatsächlich Epitope: {true_epi}")
        print(f"- Vorhergesagte Epitope: {pred_epi}")
        print("Confusionsmatrix (TN, FP / FN, TP):")
        print(cm)
        print("------------------------------------------------\n")

    # Optional: Ergebnisse zurückgeben, falls sie anderswo weiterverarbeitet werden sollen
    return (amino_acid_counts_total,
            amino_acid_counts_epitope_true,
            amino_acid_counts_epitope_predicted,
            confusion_matrices)
