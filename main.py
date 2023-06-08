
# import sonnet as snt
import tensorflow as tf
import keras
# import tensorflow_cloud as tfc
from tensorflow import keras
from keras import regularizers
import pandas as pd
import numpy as np

from sklearn.utils import class_weight

from matplotlib import pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding

from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras.layers.core.dropout import Dropout
from keras.layers import Flatten
import keras_metrics
from keras.models import load_model

import keras.optimizers as opt
from keras.preprocessing.text import one_hot
from keras import backend as K


def read_data(filepath):
    df = pd.read_excel(filepath, skiprows = [-1])

    sequence_as_aminoacids_list: list = []

    first_col: str = 'Epitope'
    epitope_embed_list: list = []

    for i, sequence in enumerate(df['Sequence']):

        column = 3

        char_list: list = []

        sequence_as_sentence: str = ""

        epitope_embed: list = []

        loc_column = first_col

        for char in str(sequence):
            char_list.append(char)
            sequence_as_sentence += char + " "

            epitope_embed.append(0)

        while column < 234:

            epitope_encoded: list = str(df.loc[i, loc_column]).replace(" ", "").split(",")

            for yeetitope in epitope_encoded:

                if yeetitope != "nan":
                    # print('Yeetitope: ' + str(yeetitope[1:]))
                    epitope_embed[int(yeetitope[1:]) - 1] = 1

            loc_column = 'Unnamed: ' + str(column)
            column += 1

        epitope_embed_list.append(epitope_embed)
        if epitope_embed.count(1) < 1:
            print(epitope_embed.count(1))

        # sequence_as_aminoacids_list.append(char_list)
        sequence_as_aminoacids_list.append(sequence_as_sentence)
    # print(epitope_embed_list)
    return sequence_as_aminoacids_list, epitope_embed_list


def embedding(filepath):
    sequence_list, epitope_embed_list = read_data(filepath)
    print('sequence_list: ' + str(len(sequence_list)))

    voc_size = 100

    length_of_longest_sequence = int(len(max(sequence_list, key = len)) / 2)

    # embedded_docs = pad_sequences(onehot_repr, padding='post', maxlen=(length_of_longest_sequence))
    # print(embedded_docs)

    # print(length_of_longest_sequence)
    # embedded_docs = pad_sequences(onehot_repr, padding='post', maxlen=(length_of_longest_sequence))

    encoder = keras.preprocessing.text.Tokenizer()

    encoder.fit_on_texts(sequence_list)

    pre_embedded_docs = encoder.texts_to_sequences(sequence_list)
    embedded_docs = keras.preprocessing.sequence.pad_sequences(pre_embedded_docs, maxlen = length_of_longest_sequence,
                                                               padding = 'post')

    # epitope_embed_list = keras.preprocessing.sequence.pad_sequences(epitope_embed_list, maxlen=length_of_longest_sequence, padding='post')
    embedded_docs = np.array(embedded_docs)

    max_len_antigen: int = len(max(epitope_embed_list, key = len))

    for i, epitope in enumerate(epitope_embed_list):
        x = max_len_antigen - len(epitope)
        y = 0
        while y < x:
            epitope.append(0)
            y += 1

        epitope_embed_list[i] = epitope
    embedded_docs = np.array(embedded_docs)

    return embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder


def plot_sth(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()


def custom_binary_loss(y_true, y_pred):
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon())  # Cancels out when target is 1
    term_1 = y_true * K.log(y_pred + K.epsilon())  # Cancels out when target is 0

    return -K.mean(term_0 + term_1, axis = 1)


def create_ai(filepath):

    embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder = embedding(filepath)

    optimizer = opt.adam_v2.Adam(learning_rate = 0.01)

    antigen_list = embedded_docs[:-30]
    epitope_list = epitope_embed_list[:-30]

    new_antigen_list = []
    new_epitope_list = []

    for sequence, antigen in zip(epitope_list, antigen_list):

        short_epitope: list = []
        short_antigen: list = []
        context_length = 20

        for i, aminoacid, char in enumerate(zip(sequence, antigen)):

            while (context_length > 0) and (i < (len(sequence) + 1)):

                if aminoacid == 1:

                    short_epitope.append(aminoacid)
                    short_antigen.append(char)
                    context_length = 20

                elif aminoacid == 0:
                    short_epitope.append(-1)
                    short_antigen.append(char)
                    context_length -= 1

        new_epitope_list.appendc(short_epitope)
        new_antigen_list.append(short_antigen)

    new_epitope_list = keras.preprocessing.sequence.pad_sequences(new_epitope_list, maxlen = length_of_longest_sequence,
                                                               padding = 'post')
    new_antigen_list = keras.preprocessing.sequence.pad_sequences(new_antigen_list, maxlen = length_of_longest_sequence,
                                                                  padding = 'post')


    epitope_array = np.array(new_epitope_list[:-30], dtype = np.float32)
    antigen_array = np.array(new_antigen_list[:-30], dtype = np.float32)

    test_y = np.array(new_epitope_list[-30:], dtype = np.float32)
    test_x = np.array(new_antigen_list[-30:], dtype = np.float32)
    # please add back , 1
    trainx = np.reshape(antigen_array, (antigen_array.shape[0], antigen_array.shape[1], 1))
    trainy = np.reshape(epitope_array, (epitope_array.shape[0], epitope_array.shape[1], 1))
    print(trainy.shape)
    testx = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    testy = np.reshape(test_y, (test_y.shape[0], test_y.shape[1], 1))


    single_sequence_for_testing = antigen_list[:1]
    single_epitope_to_seqeuence_for_testing = epitope_list[:1]


    embedding_dim = 3

    np.seterr(all = None, divide = None, over = 'warn', under = None, invalid = None)
    """       
    model = Sequential()

    model.add(Embedding(len(encoder.index_word) + 1, 5,
                        input_length = length_of_longest_sequence,
                        input_shape = (length_of_longest_sequence,), mask_zero = True))


    model.add(layers.Bidirectional(layers.recurrent_v2.LSTM(24, activation='tanh',
                  recurrent_activation='sigmoid',
                  input_shape=(len(antigen_list[0]), embedding_dim),
                                          return_sequences=True,
                                          dropout=0.3,
                                          bias_regularizer=None,
                                          recurrent_regularizer=None,
                                          name='LSTM')))

    model.add(layers.TimeDistributed(Dense(1, activation='sigmoid', name='output')))   
    """

    model.compile(optimizer, loss = 'binary_crossentropy',
                  metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
                  )

    history = model.fit(trainx, trainy, epochs = 2, verbose = 1
                        # , validation_split=0.2
                        , sample_weight = class_weight.compute_sample_weight('balanced', epitope_array)
                        )

    loss, accuracy, precision, recall = model.evaluate(testx, testy, verbose = 1)

    # model.save('my_model.h5')

    # plot_sth(history)

    print(model.summary())
    # print(model.predict(test_x))
    # print(test_y)
    print('Accuracy: %f' % (accuracy * 100))
    print('Loss: %f' % (loss * 100))
    print('Precision: %f' % (precision * 100))
    print('Recall: %f' % (recall * 100))




create_ai('/content/Dataset.xlsx')

