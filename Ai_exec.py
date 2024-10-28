import numpy as np
import tf_keras
from tf_keras import optimizers as opt, layers
from transformers import TFEsmModel

from colab2 import embedding, modify_with_context, calculating_class_weights, TokenAndPositionEmbedding, \
    TransformerBlock, TransformerDecoderTwo, get_weighted_loss, save_ai, use_model_and_predict, new_embedding
from validate_45_blind import validate_on_45_blind


def create_ai(filepath, save_file, output_file, train=False, safe=False,  validate=False, predict=False, old=False):
    embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder = embedding(filepath, old=old)
    print("Neue Anzahl an Sequenzen" + str(len(embedded_docs)))

    optimizer = tf_keras.optimizers.Adam(learning_rate = 0.001)
    # optimizersgd = opt.sgd_experimental.SGD(learning_rate=0.001, clipnorm=5)

    antigen_list = embedded_docs[:-300]
    epitope_list = epitope_embed_list[:-300]

    testx_list = embedded_docs[-300:]
    testy_list = epitope_embed_list[-300:]

    antigen_list_full_sequence = antigen_list
    epitope_list_full_sequence = epitope_list

    print(antigen_list[0])

    epitope_list, antigen_list, length_of_longest_context = modify_with_context(epitope_list, antigen_list,
                                                                                length_of_longest_sequence)
    print(antigen_list)
    testy_list, testx_list, length_of_longest_context_2 = modify_with_context(testy_list, testx_list,
                                                                              length_of_longest_sequence)
    if old==False:
        antigen_list = new_embedding(antigen_list, encoder)
        testx_list = new_embedding(testx_list, encoder)

    epitope_list_for_weights = epitope_list
    epitope_list_for_weights = np.array(epitope_list_for_weights, dtype = np.float32)

    epitope_list = np.array(epitope_list, dtype = np.float32)
    antigen_list = np.array(antigen_list, dtype = np.float32)

    epitope_list_for_weights = np.reshape(epitope_list_for_weights,
                                          (epitope_list_for_weights.shape[0], epitope_list_for_weights.shape[1]))
    epitope_list = np.reshape(epitope_list, (epitope_list.shape[0], epitope_list.shape[1], 1))
    antigen_list = np.reshape(antigen_list, (antigen_list.shape[0], antigen_list.shape[1], 1))

    testy_list_for_weights = np.array(testy_list, dtype = np.float32)
    testy_list = np.array(testy_list, dtype = np.float32)
    testx_list = np.array(testx_list, dtype = np.float32)

    testy_for_weights = np.reshape(testy_list_for_weights,
                                   (testy_list_for_weights.shape[0], testy_list_for_weights.shape[1]))
    testy_list = np.reshape(testy_list, (testy_list.shape[0], testy_list.shape[1], 1))
    testx_list = np.reshape(testx_list, (testx_list.shape[0], testx_list.shape[1], 1))

    print("EPITOPE", epitope_list.shape, "ANTIGEN", antigen_list.shape)

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
    single_sequence_for_testing = antigen_list[:1]
    single_epitope_to_seqeuence_for_testing = epitope_list[:1]

    # weights = class_weight.compute_sample_weight(class_weight='balanced', y=epitope_array)
    # print(pd.Series(test_sample_weights).unique())
    embedding_dim = 4
    # model = load_model('/my_test_model_02(1).h5', compile=False)

    np.seterr(all = None, divide = None, over = 'warn', under = None, invalid = None)

    num_transformer_blocks = 2
    num_decoder_blocks = 1
    embed_dim = 320  # Embedding size for each token used to be 24
    num_heads = 40  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    maxlen = length_of_longest_context
    rate = 0.1
    training = True


    some_class_weight = {0: 1.,
                         1: 3.}


    if train:

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

        embedding_layer = TokenAndPositionEmbedding(maxlen, voc_size, embed_dim)
        if old:
            encoder_embed_out = embedding_layer(encoder_inputs)
            x = encoder_embed_out
        else:
            esm_model = TFEsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D')
            esm_outputs = esm_model(encoder_inputs)['last_hidden_state']
            x = esm_outputs
            print("Shape of esm_outputs:", esm_outputs.shape)
            output_dimension = esm_outputs.shape[2]
        for i in range(num_transformer_blocks):
            transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
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

        history = model.fit(x = antigen_list, y = epitope_list, batch_size = 16, epochs = 100,
                            validation_data = (testx_list, testy_list), callbacks = [callback], verbose=1)
        # history = model.fit(x=antigen_list, y=epitope_list, batch_size=50, epochs=100, validation_data=(testx_list, testy_list, testy_for_weights), callbacks=[callback], sample_weight = epitope_list_for_weights)

        # plot_results(history)



        tf_keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
                                  to_file = '/content/multi_model' + str(i) + '.png')
        if safe:
            save_ai(model, save_file)
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
