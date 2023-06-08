from pandas.io.pytables import AppendableMultiSeriesTable

from typing_extensions import ParamSpecArgs
from tensorflow.python.eager.context import ContextSwitch

import statistics
import math
# Load the Drive helper and mount
from google.colab import drive
# drive.mount('/content/drive')


import tensorflow as tf
from tensorflow import keras
import pickle

# from keras.metrics import FalsePositives
print('start')
from keras.layers.core.dropout import Dropout
# import sonnet as snt

# import tensorflow_cloud as tfc

from keras import regularizers
import pandas as pd
import numpy as np
from numpy import inf

from keras import Model

from functools import partial

from sklearn.utils import class_weight

from matplotlib import pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from keras.layers import Embedding

from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras.layers import Flatten, GlobalAveragePooling2D
import keras_metrics
from keras.models import load_model

import keras.optimizers as opt
from keras.preprocessing.text import one_hot
from keras import backend as K

from keras.regularizers import l2
from keras.regularizers import l1

import datetime
import random

from scipy.stats import gmean

from scipy.stats import entropy

from numpy import mean
from numpy import average

import keras_nlp

from keras_nlp.layers import TokenAndPositionEmbedding as TAPE
from keras_nlp.layers import TransformerEncoder as TE

random.seed(10)


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

    voc_size = 100

    length_of_longest_sequence = int(len(max(sequence_list, key = len)) / 2)

    encoder = keras.preprocessing.text.Tokenizer(num_words = 35, char_level = True)

    # loading

    with open('/content/drive/MyDrive/ifp/tokenizer(1).pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    encoder.fit_on_texts(sequence_list)

    # saving
    """
    with open('tokenizer.pickle', 'wb') as handle:
      pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """

    pre_embedded_docs = encoder.texts_to_sequences(sequence_list)
    embedded_docs = keras.preprocessing.sequence.pad_sequences(pre_embedded_docs, maxlen = length_of_longest_sequence,
                                                               padding = 'post', value = 0)

    epitope_embed_list = keras.preprocessing.sequence.pad_sequences(epitope_embed_list,
                                                                    maxlen = length_of_longest_sequence,
                                                                    padding = 'post', value = 0)
    # embedded_docs = np.array(embedded_docs)

    max_len_antigen: int = len(max(epitope_embed_list, key = len))

    # embedded_docs = np.array(embedded_docs)

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


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim = 256, num_heads = 4, ff_dim = 32, rate = 0., **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation = "relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon = 1e-6)

        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output, weights = self.att(inputs, inputs, return_attention_scores = True)
        attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        return self.layernorm2(out1 + ffn_output)

    """
    def build(self, input_shape):
            self.att._build_from_signature(4, 32)
    """

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            # 'att': self.att,
            # 'ffn': self.ffn,
            # 'layernorm1': self.layernorm1,
            # 'layernorm2': self.layernorm2,
            # 'dropout1': self.dropout1,
            # 'dropout2': self.dropout2,
        })
        return config


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen = 1000, vocab_size = 100, embed_dim = 40, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim = vocab_size, output_dim = embed_dim, mask_zero = True)
        self.pos_emb = layers.Embedding(input_dim = maxlen, output_dim = embed_dim, mask_zero = True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start = 0, limit = maxlen, delta = 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            # 'token_emb': self.token_emb,
            # 'pos_emb': self.pos_emb
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TokenAndPositionEmbedding2(layers.Layer):
    def __init__(self, maxlen = 1000, vocab_size = 100, embed_dim = 40, **kwargs):
        super(TokenAndPositionEmbedding2, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(input_dim = vocab_size, output_dim = embed_dim)
        self.pos_emb = layers.Embedding(input_dim = maxlen, output_dim = embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start = 0, limit = maxlen, delta = 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            # 'token_emb': self.token_emb,
            # 'pos_emb': self.pos_emb
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = embed_dim,
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim = embed_dim,
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(latent_dim, activation = "relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask = None):
        """
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        """
        attention_output_1 = self.attention_1(
            query = inputs, value = inputs, key = inputs,  # attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query = out_1,
            value = encoder_outputs,
            key = encoder_outputs,
            # attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype = "int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype = tf.int32)],
            axis = 0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        # ['self', 'embed_dim', 'latent_dim', 'num_heads']
        config = super().get_config().copy()
        config.update({
            # 'self': self,
            # 'embed_dim': self.embed_dim,
            # 'latent_dim': self.latent_dim,
            # 'num_heads': self.num_heads,
            # 'dropout1': self.dropout1,
            # 'dropout2': self.dropout2,
        })
        return config


def custom_loss_4(y_true, y_pred, weights):
    return K.mean(K.abs(y_true - y_pred) * weights)


def custom_binary_loss(y_true, y_pred):
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    term_0 = (1 - y_true) * K.log(1 - y_pred + K.epsilon()) * 3.821061354e-3  # Cancels out when target is 1
    term_1 = y_true * K.log(y_pred + K.epsilon()) * 0.9961789386  # Cancels out when target is 0

    return -K.mean(term_0 + term_1, axis = 1)


def my_loss(weight):
    def weighted_cross_entropy_with_logits(labels, logits):
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels, logits, weight
        )
        return loss

    return weighted_cross_entropy_with_logits


def my_loss_2(y_true, y_pred):
    """(sum([(t-p)**2 for t,p in zip(y_true, y_pred)])/n_nonzero)**0.5"""

    return K.sqrt(K.sum(K.square(y_pred * K.cast(y_true > 0, "float32") - y_pred), axis = -1) / K.sum(
        K.cast(y_true > 0, "float32")))


def modify_with_context(epitope_list, antigen_list, length_of_longest_sequence):
    """ The sequences are going to be cut into shorter pieces, where the first aminoacid being part of an epitope is marked as the start.
        A random number of non-epitope aminoacids will be added infront of the starting epitope.

        context: defines the length after which the sequence will be cut if no epitope was found.

        returns the new antigen and epitope list aswell as the new length of the longest sequence to which every new sequence is padded."""
    new_antigen_list: list = []
    new_epitope_list: list = []
    context = 20
    for sequence, antigen in zip(epitope_list, antigen_list):

        short_epitope: list = []
        short_antigen: list = []
        context_length = 0
        i = 0
        start = True

        for run, (aminoacid, char) in enumerate(zip(sequence, antigen)):
            i += 1
            if aminoacid == 1:

                if start == True:

                    number = random.randint(2, context)

                    while number > 0:
                        short_epitope.append(0)
                        short_antigen.append(antigen[run - number])
                        number -= 1

                start = False
                short_epitope.append(1)
                short_antigen.append(char)
                context_length = context

            elif context_length < 1:

                continue

            elif (aminoacid == 0) and (i < length_of_longest_sequence + 1) and (context_length > 0):
                short_epitope.append(0)
                short_antigen.append(char)
                context_length -= 1

        new_epitope_list.append(short_epitope)
        new_antigen_list.append(short_antigen)

    length_of_longest_context = int(len(max(new_antigen_list, key = len)))
    # print(short_epitope)

    new_epitope_list = keras.preprocessing.sequence.pad_sequences(new_epitope_list, maxlen = length_of_longest_context,
                                                                  padding = 'post', value = 0)
    new_antigen_list = keras.preprocessing.sequence.pad_sequences(new_antigen_list, maxlen = length_of_longest_context,
                                                                  padding = 'post', value = 0)

    return new_epitope_list, new_antigen_list, length_of_longest_context


def create_ai(filepath):
    embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder = embedding(filepath)

    # optimizer = opt.adam_v2.Adam(learning_rate=0.1)

    optimizer = opt.adam_v2.Adam(learning_rate = 0.001)
    # optimizersgd = opt.sgd_experimental.SGD(learning_rate=0.001, clipnorm=5)

    antigen_list = embedded_docs[:-30]
    epitope_list = epitope_embed_list[:-30]

    testx_list = embedded_docs[-30:]
    testy_list = epitope_embed_list[-30:]

    antigen_list_full_sequence = antigen_list
    epitope_list_full_sequence = epitope_list

    print(antigen_list)
    epitope_list, antigen_list, length_of_longest_context = modify_with_context(epitope_list, antigen_list,
                                                                                length_of_longest_sequence)
    print(antigen_list)

    antigen_list, epitope_list, epitope_list2 = prepare_training_data(antigen_list, epitope_list)

    # epitope_array_2 = np.array(epitope_list_full_sequence, dtype=np.float32)
    # antigen_array_2 = np.array(antigen_list_full_sequence, dtype=np.float32)

    test_y = np.array(testy_list, dtype = np.float32)
    test_x = np.array(testx_list, dtype = np.float32)
    ### 2D shape

    # trainx = np.reshape(antigen_array, (antigen_array.shape[0], antigen_array.shape[1]))
    # trainx = antigen_array
    # trainy = epitope_array_2
    # trainy = np.reshape(epitope_array_2, (epitope_array.shape[0], epitope_array.shape[1]))

    # original 3D trainx shape
    # trainx2 = np.reshape(antigen_array, (antigen_array.shape[0], antigen_array.shape[1], 1))

    # trainy2 = np.reshape(epitope_array, (epitope_array.shape[0], epitope_array.shape[1], 1))

    # trainx_2 = np.reshape(antigen_array_2, (antigen_array_2.shape[0], antigen_array_2.shape[1]))
    # trainy_2 = np.reshape(epitope_array_2, (epitope_array_2.shape[0], epitope_array_2.shape[1]))
    testx = np.reshape(test_x, (test_x.shape[0], test_x.shape[1]))
    testy = np.reshape(test_y, (test_y.shape[0], test_y.shape[1]))

    single_sequence_for_testing = antigen_list[:1]
    single_epitope_to_seqeuence_for_testing = epitope_list[:1]

    # weights = class_weight.compute_sample_weight(class_weight='balanced', y=epitope_array)
    # print(pd.Series(test_sample_weights).unique())
    print("hi: " + str(len(encoder.index_word)))
    embedding_dim = 4
    # model = load_model('/my_test_model_02(1).h5', compile=False)

    np.seterr(all = None, divide = None, over = 'warn', under = None, invalid = None)

    num_transformer_blocks = 4
    embed_dim = 256  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    maxlen = length_of_longest_context
    print(maxlen)
    yes_do = True

    """cause my code looks like trash"""

    # model = load_model('/content/drive/MyDrive/ifp/model_random_test.h5', custom_objects={'TransformerBlock': TransformerBlock, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding}, compile=True)

    if yes_do == True:

        with tpu_strategy.scope():  # creating the model in the TPUStrategy scope means we will train the model on the TPU

            encoder_inputs = layers.Input(shape = [length_of_longest_context, ])

            embedding_layer = TokenAndPositionEmbedding(maxlen, voc_size, embed_dim)
            x = embedding_layer(encoder_inputs)

            for i in range(num_transformer_blocks):
                transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
                x = transformer_block(x)

            encoder_outputs = layers.Dense(256, activation = "sigmoid")(x)

            encoder_model = keras.Model(inputs = encoder_inputs, outputs = encoder_outputs)
            encoder_outputs = encoder_model(encoder_inputs)

            decoder_inputs = layers.Input(shape = [length_of_longest_context, ])
            encoded_seq_inputs = keras.Input(shape = (None, embed_dim,), name = "decoder_state_inputs")

            decoder_embed = TokenAndPositionEmbedding2(maxlen, voc_size, embed_dim)(decoder_inputs)

            decoder_output = TransformerDecoder(embed_dim, ff_dim, num_heads)(decoder_embed, encoded_seq_inputs)

            decoder_outputs = layers.Flatten()(decoder_output)

            decoder_outputs = layers.Dense(1, activation = "sigmoid")(decoder_outputs)

            decoder_model = keras.Model(inputs = [decoder_inputs, encoded_seq_inputs], outputs = decoder_outputs)

            decoder_outputs = decoder_model([decoder_inputs, encoder_outputs])

            transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
            # decoder_output = decoderModel([trainy, trainx])

            # model = load_model('/content/drive/MyDrive/ifp/model_context_20_01(2).h5', custom_objects={'TransformerBlock': TransformerBlock, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding}, compile=True)

            """
            model = Sequential()
  
            model.add(Embedding(len(encoder.index_word) + 1, embedding_dim,
                              input_length=None,
  
            model.add(layers.recurrent_v2.LSTM(24, activation='tanh',
                      recurrent_activation='sigmoid',
                      input_shape=(None, embedding_dim + 1 ),
                                              return_sequences=True, 
                                              dropout=0.,
                                              bias_regularizer=None,
                                              recurrent_regularizer=None,
                                              name='LSTM'))
  
            model.add(layers.TimeDistributed(Dense(1, activation='sigmoid', name='output')))
  
            """
            for i, sequence in enumerate(antigen_list):

                epitope_array = np.array(epitope_list[i], dtype = np.float32)
                epitope_array_2 = np.array(epitope_list2[i], dtype = np.float32)
                antigen_array = np.array(sequence, dtype = np.float32)

                trainx2 = np.reshape(antigen_array, (antigen_array.shape[0], antigen_array.shape[1], 1))
                trainy2 = np.reshape(epitope_array, (epitope_array.shape[0], epitope_array.shape[1], 1))
                trainy = np.reshape(epitope_array_2, (epitope_array_2.shape[0], epitope_array_2.shape[1]))

                try:
                    model = load_model('/model_new_test_03.h5', custom_objects = {'TransformerBlock': TransformerBlock,
                                                                                  'TokenAndPositionEmbedding': TokenAndPositionEmbedding},
                                       compile = True)

                except Exception:

                    transformer.compile(optimizer,
                                        # loss=my_loss_2,
                                        loss = 'binary_crossentropy',
                                        metrics = ['accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
                                        # sample_weight_mode='temporal'
                                        )

                history = transformer.fit([trainx2, trainy2], trainy, batch_size = 128, epochs = 100, verbose = 1
                                          # ,sample_weight=weights
                                          )

                transformer.save('/model_new_test_03.h5')

            # loss, accuracy, precision, recall = model.evaluate(testx, testy, verbose=1)

            # prediction = model.predict(x=trainx)

            print(transformer.summary())

            # new_function([testx], testy, length_of_longest_context, length_of_longest_sequence, transformer)

            # print(len(antigen_list[0]))
            # print("Prediction for x: " + str(prediction))
            # print(len(prediction))
            # print(trainx[0])
            # print("trainy: " + str(trainy_2[1]))

            # plot_sth(history)

            print(transformer.summary())
            # print(model.predict(test_x))
            # print(test_y)
            # print('Accuracy: %f' % (accuracy * 100))
        # print('Loss: %f' % (loss * 100))
        # print('Precision: %f' % (precision * 100))
        # print('Recall: %f' % (recall * 100))

        # print('testx_shape: ' + str(testx.shape))

        # load_model_and_do_stuff([trainx2[0], trainy2[0]], trainy, maxlen, voc_size, embed_dim, transformer, length_of_longest_context)


def new_function(testx, testy, length_of_longest_context, length_of_longest_sequence, model = None):
    new_x, count = split_sequences(testx, testy, length_of_longest_context)

    for i, x in enumerate(new_x):
        new_x = x
        new_x = np.array(new_x, dtype = np.float32)

        new_values = modified_prediction(new_x, count, length_of_longest_sequence, model)
        # print(new_values)
        new_values = np.array(new_values, dtype = np.float32)
        plot_this(new_values, testy[i])


def plot_this(prediction, testy):
    # print(prediction)
    plt.figure(figsize = (10, 10))
    plt.plot(prediction)
    plt.show()

    plt.figure(figsize = (10, 10))
    plt.plot(testy)
    plt.ylim((0., 1.0))
    plt.show()


def split_sequences(x, y, length_of_longest_context):
    new_sequences_x: list = []
    new_sequences_y: list = []
    total_amount_from_one_sequence: int = 0

    for counter in range(30):
        new_sequences_x.append([])

    for i, sequence in enumerate(x):
        print(len(sequence))

        for j in range(len(sequence) - length_of_longest_context):
            new_sequences_x[i].append(sequence[j:(length_of_longest_context + j)])

            if j > total_amount_from_one_sequence:
                total_amount_from_one_sequence = j

    return new_sequences_x, total_amount_from_one_sequence


def modified_prediction(new_sequences_x, count, length_of_full_sequence, model = None):
    if model is None:
        model = load_model('/model_new_test_03.h5', custom_objects = {'TransformerBlock': TransformerBlock,
                                                                      'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                                                      'TransformerDecoder': TE}, compile = True)
    prediction = model.predict(x = new_sequences_x, verbose = 1)
    new_predictions: list = []

    print(prediction.shape)

    for i in range(length_of_full_sequence):
        new_predictions.append([])

    print(prediction.shape)

    for i in range(count):
        for j in range(len(prediction[i])):
            if ((prediction[i, j, 0] > 0.001)):
                new_predictions[i + j].append(prediction[i, j, 0])

    geometric_mean_for_new_prediction: list = []

    for z in range(len(new_predictions)):

        value = gmean(new_predictions[z], axis = None)

        if math.isnan(value):
            value = np.array(0., dtype = float)

        geometric_mean_for_new_prediction.append(value.item())

    # print("länge der Vorhersage: " + str(len(geometric_mean_for_new_prediction)))
    return geometric_mean_for_new_prediction


def load_model_and_do_stuff(testx, testy, maxlen, voc_size, embed_dim, model, length_of_longest_context):
    # obj = TokenAndPositionEmbedding(maxlen, voc_size, embed_dim)
    # obj.__init__(maxlen, voc_size, e.h5', custom_objects={'TransformerBlock': TransformerBlock, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding}, compile=False)
    # model = load_model('/model_random_test.h5', custom_objects={'TransformerBlock': TransformerBlock, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding}, compile=True)
    # print('testx_shape: ' + str(testx.shape))
    prediction = model.predict(x = testx)

    classes_x = np.where(prediction > 0.5, 1, 0)
    print(classes_x.shape)
    # loss, accuracy, precision, recall = model.evaluate(testx, testy, verbose=1)
    # yhat = model.predict_classes(testx, verbose=0)
    """
    for i in range(testx.shape[1]):
  
          print('Expected:', testy[0, i], 'Predicted', classes_x[0, i])  
    """

    for i in range(1):
        # print(prediction)
        plt.figure(figsize = (10, 10))
        plt.plot(prediction)
        plt.plot(testy[0], 'o', alpha = 0.3, color = 'red')
        plt.ylim((0., 1.0))
        plt.show()


def prepare_training_data(trainx2, trainy2):
    new_trainx2 = []
    new_trainy2 = []
    new_trainy = []
    for counter, i in enumerate(trainy2):
        new_trainy2_parts = []
        new_trainx2_parts = []
        new_trainy_parts = []
        for j in range(len(i) - 3):
            new_trainx2_parts.append(trainx2[counter])
            new_trainy2_parts.append(i[:(j + 2)])
            new_trainy_parts.append(i[j + 3])

        new_trainx2.append([new_trainx2_parts])
        new_trainy2.append([new_trainy2_parts])
        new_trainy.append([new_trainy_parts])

    new_trainy2y = []
    new_trainx2x = []
    new_trainyy = []
    for i, item in enumerate(new_trainy2[0]):
        new_trainy2y.append([])
        new_trainx2x.append([])
        new_trainyy.append([])

        for j, jtem in enumerate(new_trainy2):
            new_trainy2y[i].append([])
            new_trainx2x[i].append([])
            new_trainyy[i].append([])

    for i, item in enumerate(new_trainy2[0]):
        for j, jtem in enumerate(new_trainy2):
            new_trainy2y[i][j].append(new_trainy2[j][i])
            new_trainx2x[i][j].append(new_trainx2[j][i])
            new_trainyy[i][j].append(new_trainy[j][i])

    return new_trainx2x, new_trainy2y, new_trainyy


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    raise BaseException(
        'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

create_ai('/content/drive/MyDrive/ifp/Dataset-without-1550.xlsx')