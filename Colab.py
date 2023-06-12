# from tensorflow.core.framework.op_def_pb2 import tensorflow_dot_core_dot_framework_dot_full__type__pb2
# from pandas.io.pytables import AppendableMultiSeriesTable

# from typing_extensions import ParamSpecArgs
# from tensorflow.python.eager.context import ContextSwitch
###########################START##############################
import os
import statistics
import math


import tensorflow as tf

print("Tensorflow version " + tf.__version__)
from tensorflow import keras
import pickle

print('Start 1')



from keras import regularizers
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from keras import layers
from keras.models import load_model
import keras.optimizers as opt
from keras import backend as K
import datetime
import random
from scipy.stats import gmean
import keras_nlp
from keras_nlp.layers import TokenAndPositionEmbedding as TAPE
from keras_nlp.layers import TransformerEncoder as TE

# set seed to counter rng during training
random.seed(10)
tf.random.set_seed(10)


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
            ### + " "
            sequence_as_sentence += char

            epitope_embed.append(-1)

        while column < 234:

            epitope_encoded: list = str(df.loc[i, loc_column]).replace(" ", "").split(",")

            for yeetitope in epitope_encoded:

                if yeetitope != "nan":
                    epitope_embed[int(yeetitope[1:]) - 1] = 1

            loc_column = 'Unnamed: ' + str(column)
            column += 1

        # if (epitope_embed.count(1) > 4) and epitope_embed.count(1) < 18 :
        if epitope_embed.count(1) > 4:
            epitope_embed_list.append(epitope_embed)
            sequence_as_aminoacids_list.append(sequence_as_sentence)

    return sequence_as_aminoacids_list, epitope_embed_list


def embedding(filepath):
    sequence_list, epitope_embed_list = read_data(filepath)

    voc_size = 100

    length_of_longest_sequence = int(len(max(sequence_list, key = len)) / 2)

    encoder = keras.preprocessing.text.Tokenizer(num_words = 35, char_level = True)

    # loading

    with open('/content/drive/MyDrive/ifp/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    encoder.fit_on_texts(sequence_list)
    print(encoder.word_index)

    pre_embedded_docs = encoder.texts_to_sequences(sequence_list)
    """
    # saving

    with open('/content/drive/MyDrive/ifp/tokenizer.pickle', 'wb') as handle:
      pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(encoder.word_index)"""
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


# class TransformerBlock(tf.keras.Model):
class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim = 256, num_heads = 4, ff_dim = 32, rate = 0.3, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    @tf.function
    def call(self, inputs, training = True, mask = None):
        attn_output = self.att(inputs, inputs, attention_mask = mask)
        if training:
            attn_output = self.dropout1(attn_output, training = training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        if training:
            ffn_output = self.dropout2(ffn_output, training = training)
        return self.layernorm2(out1 + ffn_output)

    def compute_mask(self, inputs, mask = None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(num_heads = self.num_heads, key_dim = self.embed_dim, dropout = 0.3)
        self.ffn = keras.Sequential(
            [layers.TimeDistributed(layers.Dense(self.ff_dim, activation = "relu")),
             layers.TimeDistributed(layers.Dropout(rate = self.rate)),
             layers.TimeDistributed(layers.Dense(self.embed_dim)), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon = 1e-6)

        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

    def get_config(self):

        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# class TokenAndPositionEmbedding(tf.keras.Model):
class TokenAndPositionEmbedding(tf.keras.layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def compute_mask(self, inputs, mask = None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    def build(self, input_shape):
        self.token_emb = layers.Embedding(input_dim = self.vocab_size, output_dim = self.embed_dim, mask_zero = True)
        self.pos_emb = layers.Embedding(input_dim = self.maxlen, output_dim = self.embed_dim, mask_zero = True)

    @tf.function
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start = 0, limit = maxlen, delta = 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        # config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            # 'token_emb': self.token_emb,
            # 'pos_emb': self.pos_emb
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# class TokenAndPositionEmbedding2(tf.keras.Model):
class TokenAndPositionEmbedding2(tf.keras.layers.Layer):

    def __init__(self, maxlen = 1000, vocab_size = 100, embed_dim = 40, **kwargs):
        super(TokenAndPositionEmbedding2, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.token_emb = layers.Embedding(input_dim = self.vocab_size, output_dim = self.embed_dim, mask_zero = True)
        self.pos_emb = layers.Embedding(input_dim = self.maxlen, output_dim = self.embed_dim, mask_zero = True)

    def compute_mask(self, inputs, mask = None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    @tf.function
    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start = 0, limit = maxlen, delta = 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super(TokenAndPositionEmbedding2, self).get_config()
        # config = super().get_config().copy()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,

        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# class TransformerDecoder(tf.keras.Model):
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, rate = 0.3, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.rate = rate

    def compute_mask(self, inputs, mask = None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    @tf.function
    def call(self, decoder_inputs, encoder_outputs, training = True, mask = None):

        causal_mask = self.get_causal_attention_mask(decoder_inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype = "int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(
            query = decoder_inputs,
            value = decoder_inputs,
            key = decoder_inputs,
            attention_mask = causal_mask,
        )

        if training:
            attention_output_1 = self.dropout1(attention_output_1, training = training)

        out_1 = self.layernorm_1(decoder_inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query = out_1,
            value = encoder_outputs,
            key = encoder_outputs,
            # attention_mask=mask,
            attention_mask = padding_mask
        )

        if training:
            attention_output_2 = self.dropout2(attention_output_2, training = training)

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

    def build(self, input_shape):
        self.attention_1 = layers.MultiHeadAttention(
            num_heads = self.num_heads, key_dim = self.embed_dim, dropout = 0.3, name = "att1",
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads = self.num_heads, key_dim = self.embed_dim, dropout = 0.3, name = "att2",
        )
        self.dense_proj = keras.Sequential(
            [layers.TimeDistributed(layers.Dense(self.latent_dim, activation = "relu")),
             layers.TimeDistributed(layers.Dropout(rate = self.rate)),
             layers.TimeDistributed(layers.Dense(self.embed_dim)), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'latent_dim': self.latent_dim,
            'num_heads': self.num_heads,
            'rate': self.rate

        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerDecoderTwo(tf.keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, rate = 0.3, **kwargs):
        super(TransformerDecoderTwo, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.rate = rate

    def compute_mask(self, inputs, mask = None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask

    @tf.function
    def call(self, encoder_outputs, training = True, mask = None):

        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype = "int32")
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(
            query = encoder_outputs,
            value = encoder_outputs,
            key = encoder_outputs,
            attention_mask = padding_mask,
        )

        if training:
            attention_output_1 = self.dropout1(attention_output_1, training = training)

        out_1 = self.layernorm_1(encoder_outputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query = out_1,
            value = encoder_outputs,
            key = encoder_outputs,
            # attention_mask=mask,
            attention_mask = padding_mask
        )

        if training:
            attention_output_2 = self.dropout2(attention_output_2, training = training)

        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def build(self, input_shape):
        self.attention_1 = layers.MultiHeadAttention(
            num_heads = self.num_heads, key_dim = self.embed_dim, dropout = 0.3, name = "att1",
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads = self.num_heads, key_dim = self.embed_dim, dropout = 0.3, name = "att2",
        )
        self.dense_proj = keras.Sequential(
            [layers.TimeDistributed(layers.Dense(self.latent_dim, activation = "relu")),
             layers.TimeDistributed(layers.Dropout(rate = self.rate)),
             layers.TimeDistributed(layers.Dense(self.embed_dim)), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

    def get_config(self):
        config = super(TransformerDecoderTwo, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'latent_dim': self.latent_dim,
            'num_heads': self.num_heads,
            'rate': self.rate

        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class My_Custom_Generator(keras.utils.Sequence):

    def __init__(self, x, y, labels, batch_size):
        self.x = x
        self.labels = labels
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        print(self.x.shape[0])
        return math.floor(self.x.shape[0] / self.batch_size)

        # return math.floor(self.x[0].shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_decoder_y = self.y[idx * self.batch_size: (idx + 1) * self.batch_size]

        return [batch_x, batch_decoder_y], [batch_y]


def generator_function(x, y, labels, batch_size):
    for i in range(x.shape[0]):
        batch_x = x[i * batch_size: (i + 1) * batch_size]
        batch_y = labels[i * batch_size: (i + 1) * batch_size]
        batch_decoder_y = y[i * batch_size: (i + 1) * batch_size]
        return {'encoder_inputs': np.array(batch_x, dtype = np.float32),
                'decoder_inputs': np.array(batch_decoder_y, dtype = np.float32)}, np.array(batch_y, dtype = np.float32)


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


def create_new_dataset(epitope_list, antigen_list, maxlen):
    x = antigen_list
    y = epitope_list

    new_encoder_x = []
    new_decoder_x = []
    new_y = []

    for epitope, protein in zip(epitope_list, antigen_list):

        for run, (aminoacid, char) in enumerate(zip(epitope, protein)):
            if aminoacid == 1:
                new_encoder_x.append(protein)
                new_decoder_x.append(epitope[:run])
                new_y.append(aminoacid)

    new_decoder_x = keras.preprocessing.sequence.pad_sequences(new_decoder_x, maxlen = maxlen, padding = 'post',
                                                               value = 0)

    return new_encoder_x, new_decoder_x, new_y


def modify_with_context(epitope_list, antigen_list, length_of_longest_sequence):
    """ The sequences are going to be cut into shorter pieces, where the first aminoacid being part of an epitope is marked as the start.
        A random number of non-epitope aminoacids will be added infront of the starting epitope.

        context: defines the length after which the sequence will be cut if no epitope was found.

        returns the new antigen(actually protein) and epitope list aswell as the new length of the longest sequence to which every new sequence is padded."""
    new_antigen_list: list = []
    new_epitope_list: list = []
    decoder_list: list = []
    context = 20
    for epitope, antigen in zip(epitope_list, antigen_list):

        short_epitope: list = []
        short_antigen: list = []
        context_length = 0
        i = 0
        start = True

        for run, (aminoacid, char) in enumerate(zip(epitope, antigen)):
            i += 1
            if aminoacid == 1:

                if start is True:

                    number = random.randint(0, context / 2)

                    while number > 0:
                        # short_epitope.append(-1)
                        short_epitope.append(0.)
                        short_antigen.append(antigen[run - number])
                        number -= 1

                start = False
                short_epitope.append(1.)
                short_antigen.append(char)
                context_length = context

            elif (context_length < 1) and (start is False):

                continue

            elif (aminoacid == -1) and (i < length_of_longest_sequence + 1) and (context_length > 0) and (
                    start is False):
                # short_epitope.append(-1)
                short_epitope.append(0.)
                short_antigen.append(char)
                context_length -= 1

            elif (aminoacid == 0) and (i < length_of_longest_sequence + 1) and (context_length > 0) and (
                    start is False):
                short_epitope.append(0.)
                short_antigen.append(char)
                context_length -= 1

        new_epitope_list.append(short_epitope)
        new_antigen_list.append(short_antigen)
        # print(short_antigen)
        # print(short_epitope)

    length_of_longest_context = int(len(max(new_antigen_list, key = len)))
    # print(short_epitope)
    length_of_longest_context = 235
    new_epitope_list = keras.preprocessing.sequence.pad_sequences(new_epitope_list, maxlen = length_of_longest_context,
                                                                  padding = 'post', value = 0)
    new_antigen_list = keras.preprocessing.sequence.pad_sequences(new_antigen_list, maxlen = length_of_longest_context,
                                                                  padding = 'post', value = 0)

    return new_epitope_list, new_antigen_list, length_of_longest_context


def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', classes = [0, 1], y = np.ravel(y_true, order = 'C'))
    print("WEIGHTS")
    print(weights)
    for i in range(len(weights)):
        weights[i][1] = weights[i][1] / 2.2
    print("WEIGHTS")
    print(weights)
    return weights


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
            axis = -1)

    return weighted_loss


def create_ai(filepath):
    embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder = embedding(filepath)
    print("Neue Anzahl an Sequenzen" + str(len(embedded_docs)))

    optimizer = opt.adam_v2.Adam(learning_rate = 0.001)
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
    antigen_list, epitope_list, epitope_list_train = prepare_training_data(antigen_list, epitope_list)

    test_antigen_list, test_epitope_list, test_epitope_train_list = prepare_training_data(testx_list, testy_list)

    test_epitope_array = np.array(test_epitope_list, dtype=np.float32)
    test_epitope_array_train = np.array(test_epitope_train_list, dtype=np.float32)
    test_antigen_array = np.array(test_antigen_list, dtype=np.float32)

    test_trainx2 = np.reshape(test_antigen_array, ((test_antigen_array.shape[0]* test_antigen_array.shape[2]), test_antigen_array.shape[3]))
    test_trainy2 = np.reshape(test_epitope_array, ((test_epitope_array.shape[0]* test_epitope_array.shape[2]), test_epitope_array.shape[3]))
    test_trainy = np.reshape(test_epitope_array_train, ((test_epitope_array_train.shape[0]* test_epitope_array_train.shape[2]), 1))

    np.save("/content/drive/MyDrive/ifp/test_epitope_array.npy", test_trainy2)
    np.save("/content/drive/MyDrive/ifp/test_epitope_array_train.npy", test_trainy)
    np.save("/content/drive/MyDrive/ifp/test_antigen_array.npy", test_trainx2)





    epitope_array = np.array(epitope_list, dtype=np.float32)
    epitope_array_train = np.array(epitope_list_train, dtype=np.float32)
    antigen_array = np.array(antigen_list, dtype=np.float32)

    np.save("/content/drive/MyDrive/ifp/epitope_array_half_len1.npy", epitope_array)
    np.save("/content/drive/MyDrive/ifp/epitope_array_train_half_len1.npy", epitope_array_train)
    np.save("/content/drive/MyDrive/ifp/antigen_arrayhalf_len1.npy", antigen_array)



    epitope_array_path = "/content/drive/MyDrive/ifp/epitope_array2.npy"
    epitope_array_train_path = "/content/drive/MyDrive/ifp/epitope_array_train2.npy"
    antigen_array_path = "/content/drive/MyDrive/ifp/antigen_array2.npy"

    """
    epitope_array = np.load("/content/drive/MyDrive/ifp/epitope_array_half_len1.npy")
    epitope_array_train = np.load("/content/drive/MyDrive/ifp/epitope_array_train_half_len1.npy")
    antigen_array = np.load("/content/drive/MyDrive/ifp/antigen_arrayhalf_len1.npy")
    """
    np.save("/epitope_array2.npy", epitope_array)
    np.save("/epitope_array_train2.npy", epitope_array_train)
    np.save("/antigen_array2.npy", antigen_array)
    """
    test_trainx2 = np.load("/content/drive/MyDrive/ifp/test_antigen_array.npy")
    test_trainy2 = np.load("/content/drive/MyDrive/ifp/test_epitope_array.npy")
    test_trainy = np.load("/content/drive/MyDrive/ifp/test_epitope_array_train.npy")

    epitope_array = np.load("/epitope_array2.npy")
    epitope_array_train = np.load("/epitope_array_train2.npy")
    antigen_array = np.load("/antigen_array2.npy")

    print("trainy")
    print(epitope_array_train.shape)
    print(antigen_array[0][0])
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

    ###Classweights
    new_weights = calculating_class_weights(epitope_list)

    print("trainy2, trainx2, trainy")
    print(trainy2.shape)
    print(trainx2.shape)
    print(trainy.shape)

    print("test_trainy")
    print(test_trainy.shape)
    print(test_trainx2.shape)

    print("trainy2, trainx2, trainy")
    print(test_trainy2.shape)
    print(test_trainx2.shape)
    print(test_trainy.shape)

    print(trainx2)

    unique, counts = np.unique(trainy, return_counts = True)
    print(unique, counts)
    print(np.asarray((unique, counts)).T)
    unique, counts = np.unique(test_trainy, return_counts = True)
    print(unique, counts)
    print(np.asarray((unique, counts)).T)

    single_sequence_for_testing = antigen_list[:1]
    single_epitope_to_seqeuence_for_testing = epitope_list[:1]

    # weights = class_weight.compute_sample_weight(class_weight='balanced', y=epitope_array)
    # print(pd.Series(test_sample_weights).unique())
    print("hi: " + str(len(encoder.index_word)))
    embedding_dim = 4
    # model = load_model('/my_test_model_02(1).h5', compile=False)

    np.seterr(all = None, divide = None, over = 'warn', under = None, invalid = None)

    num_transformer_blocks = 2
    num_decoder_blocks = 0
    embed_dim = 24  # Embedding size for each token
    num_heads = 40  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    maxlen = length_of_longest_context
    rate = 0.1
    print(maxlen)
    training = True


    some_class_weight = {0: 1.,
                         1: 3.}

    do_something = False
    if do_something:

        # with tpu_strategy.scope(): # creating the model in the TPUStrategy scope means we will train the model on the TPU
        callback = keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            min_delta = 0,
            patience = 10,
            verbose = 0,
            mode = 'auto',
            baseline = None,
            restore_best_weights = True)

        encoder_inputs = layers.Input(shape = (length_of_longest_context,), name = 'encoder_inputs')

        embedding_layer = TokenAndPositionEmbedding(maxlen, voc_size, embed_dim)
        encoder_embed_out = embedding_layer(encoder_inputs)

        x = encoder_embed_out
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

        model = keras.Model(inputs = encoder_inputs, outputs = decoder_outputs_final)

        model.compile(optimizer, loss = get_weighted_loss(new_weights),
                      weighted_metrics = ['accuracy', tf.keras.metrics.AUC(), keras.metrics.Precision(),
                                          keras.metrics.Recall()])
        # model.compile(optimizer, loss="binary_crossentropy", weighted_metrics=['accuracy', tf.keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])

        history = model.fit(x = antigen_list, y = epitope_list, batch_size = 50, epochs = 100,
                            validation_data = (testx_list, testy_list), callbacks = [callback])
        # history = model.fit(x=antigen_list, y=epitope_list, batch_size=50, epochs=100, validation_data=(testx_list, testy_list, testy_for_weights), callbacks=[callback], sample_weight = epitope_list_for_weights)

        # plot_results(history)

        """
        encoder_inputs = layers.Input(shape=(length_of_longest_context,), name='encoder_inputs')

        embedding_layer = TokenAndPositionEmbedding(maxlen, voc_size, embed_dim)
        encoder_embed_out = embedding_layer(encoder_inputs)

        #x = layers.TimeDistributed(layers.Dense(256, activation="relu"))(encoder_inputs)
        x = encoder_embed_out
        for i in range(num_transformer_blocks):
          transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, rate)
          x = transformer_block(x, training=training)

        x = layers.Dropout(rate=rate)(x)
        encoder_outputs_final = layers.Dense(embed_dim, activation="sigmoid")(x)


        #encoder_model = load_model('/model_new_test.h5', custom_objects={'TransformerBlock': TransformerBlock, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding}, compile=True)
        encoder_model = keras.Model(inputs = encoder_inputs, outputs = encoder_outputs_final)
        ######encoder_outputs = encoder_model(encoder_inputs)
        #encoder_outputs = layers.Dense(embed_dim, activation="sigmoid")(encoder_outputs)

        decoder_inputs = layers.Input(shape=(None,), name='decoder_inputs')         
        encoded_seq_inputs = keras.Input(shape=(None, embed_dim,), name="decoder_state_inputs")


        decoder_embed_layer = TokenAndPositionEmbedding2(maxlen, voc_size, embed_dim)
        decoder_embed = decoder_embed_layer(decoder_inputs)
        decoder_embed_out = layers.Dense(embed_dim, activation="sigmoid", name='Decoder_Embed_Sigmoid')(decoder_embed)

        decoder_outputs = TransformerDecoder(embed_dim, ff_dim, num_heads)(decoder_embed_out, encoder_outputs_final, training=training)

        #decoder_outputs = TransformerDecoder(embed_dim, ff_dim, num_heads)(decoder_embed_out, encoded_seq_inputs, training=training)

        for i in range(num_decoder_blocks):
          transformer_decoder = TransformerDecoder(embed_dim, ff_dim, num_heads)
          decoder_outputs = transformer_decoder(decoder_outputs, encoder_outputs_final, training=training)

        #decoder_outputs = layers.Dense(1, activation="sigmoid", name='Sigmoid')(decoder_outputs)
        decoder_outputs = layers.GlobalAveragePooling1D()(decoder_outputs)
        decoder_outputs = layers.Dropout(rate=rate)(decoder_outputs)
        decoder_outputs_final = layers.Dense(1, activation="sigmoid", name='Final_Sigmoid')(decoder_outputs)

        #decoder_model = keras.Model(inputs=[decoder_inputs, encoded_seq_inputs], outputs=decoder_outputs_final)
        #decoder_outputs_from_model = decoder_model([decoder_inputs, encoder_outputs_final])

        transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs_final)
        #decoder_output = decoderModel([trainy, trainx])


        #model = load_model('/content/drive/MyDrive/ifp/model_context_20_01(2).h5', custom_objects={'TransformerBlock': TransformerBlock, 'TokenAndPositionEmbedding': TokenAndPositionEmbedding}, compile=True)


        print("4")
        transformer.compile(optimizer, 
                      #loss=get_weighted_loss(new_weights),
                      loss='binary_crossentropy',
                      #metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()],
                      metrics=['accuracy', tf.keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()]
                      #sample_weight_mode='temporal'
                      )


        """
        """
        dataset = tf.data.Dataset.from_generator(generator, output_types=({"encoder_inputs": tf.int16, "decoder_inputs": tf.int16}, tf.int16),
                                                 output_shapes=({'encoder_inputs': tf.TensorShape([None, None,]),
                                                                'decoder_inputs': tf.TensorShape([None, None])
                                                                },
                                                                tf.TensorShape([None, None]))
                                                 )
        """
        """
        transformer.fit([trainx2, trainy2], trainy, 
                                  #steps_per_epoch=19923,
                                  batch_size=200,
                                  epochs=1000,
                                  verbose=1,
                                  validation_data=([test_trainx2, test_trainy2], test_trainy)
                                  )
        """

        tf.keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
                                  to_file = '/content/multi_model' + str(i) + '.png')

        # load_model_and_do_stuff(testx_list, testy_list, model)
        # load_model_and_do_stuff(antigen_list, epitope_list, model)

        """
        model.save_weights('/content/drive/MyDrive/ifp/final_AI_weights')
        model.save_weights('/content/final_AI_weights')
        model.save('/content/final_AI')
        model.save('/content/drive/MyDrive/ifp/final_AI')
        use_model_and_predict()
        """

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


def plot_results(history):
    print(history.history.keys())
    x = "2"

    # summarize history for accuracy
    plt.figure(dpi = 2500)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc = 'upper right')
    plt.show()
    # summarize history for loss
    plt.figure(dpi = 2500)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc = 'upper right')
    plt.show()

    # summarize history for recall
    plt.figure(dpi = 2500)
    plt.plot(history.history['recall_' + x])
    plt.plot(history.history['val_recall_' + x])
    plt.title('Model Recall')
    plt.ylabel('Recall')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc = 'lower right')
    plt.show()
    # summarize history for precision
    plt.figure(dpi = 2500)
    plt.plot(history.history['precision_' + x])
    plt.plot(history.history['val_precision_' + x])
    plt.title('Model Precision')
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc = 'lower right')
    plt.show()
    # summarize history for AUC-ROC
    plt.figure(dpi = 2500)
    plt.plot(history.history['auc_' + x])
    plt.plot(history.history['val_auc_' + x])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc = 'lower right')
    plt.show()


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

        for j in range(len(sequence) - length_of_longest_context):
            new_sequences_x[i].append(sequence[j:(length_of_longest_context + j)])

            if j > total_amount_from_one_sequence:
                total_amount_from_one_sequence = j

    return new_sequences_x, total_amount_from_one_sequence


def transformer_evaluation_loop(testx, test_decoderx, testy):
    for i in range(5):
        tf.keras.backend.clear_session()
        model = load_model('/content/drive/MyDrive/ifp/ki_für_tests_without_decoder_MODEL',
                           custom_objects = {'TransformerBlock': TransformerBlock,
                                             'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                             'TokenAndPositionEmbedding2': TokenAndPositionEmbedding2,
                                             'TransformerDecoder': TransformerDecoder},
                           compile = True
                           )
        model.load_weights('/content/drive/MyDrive/ifp/ki_für_tests_without_decoder_MODEL_weights')
        score = model.evaluate(x = [testx, test_decoderx], y = testy, batch_size = 50)
        print("FINAL SCORE")
        print(score)
        tf.keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
                                  to_file = '/content/multi_model' + str(i) + '.png')

    print(model.summary(expand_nested = True))


def split_string(input_string: str):
    if len(input_string) <= 235:
        # If the input string is shorter than 232 characters, return it as a single element in a list
        return [input_string]
    else:
        # If the input string is longer than 232 characters, split it into substrings of length 232
        # and return those substrings in a list
        substrings = []
        start_index = 0
        while start_index < len(input_string):
            end_index = start_index + 235
            if end_index > len(input_string):
                end_index = len(input_string)
            substrings.append(input_string[start_index:end_index])
            start_index += 235
        return substrings


##########################################################################################
def use_model_and_predict():
    """Enter a sequence to use for prediction and generate the heatmap output.
    All path need to be changed to wherever the files are stored on your computer."""
    sequence = "tpenitdlcaeyhntqihtlnnkifsyteslagkremaiitfkdgatfevevpgsehidsekkaiermkdtlriaylteakveklcvwnnktphaiaaisman"  # Hier die Sequenz eingeben#
    tf.keras.backend.clear_session()
    """change the following path to the final_AI folder path"""
    model = load_model('G:/Users/tinys/PycharmProjects/teststuff/AI/final_AI',
                       custom_objects = {'TransformerBlock': TransformerBlock,
                                         'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                         'TransformerDecoder': TransformerDecoder, "weighted_loss": get_weighted_loss},
                       compile = True
                       )
    """change the following path to path/final_AI_weights """
    model.load_weights('G:/Users/tinys/PycharmProjects/teststuff/AI/final_AI_weights')

    sequence_list = split_string(sequence)

    """change the following path accordingly"""
    with open('G:/Users/tinys/PycharmProjects/teststuff/AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    print(encoder.word_index)

    pre_embedded_docs = encoder.texts_to_sequences(sequence_list)
    embedded_docs = keras.preprocessing.sequence.pad_sequences(pre_embedded_docs, maxlen = 235, padding = 'post',
                                                               value = 0)

    predictions = model.predict(embedded_docs)
    x = 1
    pred_list = []
    for i, (pred, seq) in enumerate(zip(predictions, sequence)):
        for (j, seq2) in zip(pred, sequence):
            pred_list.append(j)
            print(str(x) + ": " + str(j) + " - " + str(sequence[x - 1]))
            x += 1

    create_heatmap(pred_list, sequence)


def create_heatmap(data, sequence):
    """Input: predictions from the model
    Output: Heatmaps according to the predictions for the whole sequence entered"""
    data = np.array(data[:len(sequence)], dtype = np.float32)
    # data = np.reshape(data, (data.shape[1], data.shape[0]))

    data_list, sequence_list = create_blocks(data, sequence)
    print(data_list, sequence_list)

    for i, (pred, seq) in enumerate(zip(data_list, sequence_list)):
        filename = "G:/Users/tinys/PycharmProjects/teststuff/AI/pictures/" + str(i) + ".png"
        pred = np.reshape(pred, (pred.shape[1], pred.shape[0]))
        plt.figure(dpi = 2500)
        heat_map = sb.heatmap(pred, xticklabels = seq, yticklabels = False, vmin = 0.2, vmax = 0.8, cmap = "rocket_r")
        plt.savefig(filename, dpi = 2500, bbox_inches = "tight")
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
        blocks1.append(list1[start:end])

    if len(list1) % block_size > 0:
        start = num_blocks1 * block_size
        end = len(list1)
        blocks1.append(list1[start:end])

    for i in range(num_blocks2):
        start = i * block_size
        end = start + block_size
        blocks2.append(list2[start:end])

    if len(list2) % block_size > 0:
        start = num_blocks2 * block_size
        end = len(list2)
        blocks2.append(list2[start:end])

    return blocks1, blocks2

###################################################################################
def transformer_prediction_loop(testx, test_decoder_x, testy):
    tf.keras.backend.clear_session()
    model = load_model('/content/drive/MyDrive/ifp/ki_für_tests_without_decoder_MODEL',
                       custom_objects = {'TransformerBlock': TransformerBlock,
                                         'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                         'TokenAndPositionEmbedding2': TokenAndPositionEmbedding2,
                                         'TransformerDecoder': TransformerDecoder},
                       compile = True
                       )
    model.load_weights('/content/drive/MyDrive/ifp/ki_für_tests_without_decoder_MODEL_weights')
    print('DECODER_START')
    print(test_decoder_x[0])
    test_x_input = testx[:1]
    print(test_x_input.shape)
    print(test_x_input)
    print(testx[0])
    print(testx[1])

    partial_result_list = test_decoder_x[:1]

    print(partial_result_list.shape)
    for i in range(len(testx[0]) - 2):
        actual_decoder = test_decoder_x[i:(i + 1)]
        print(actual_decoder)
        test_x_input = testx[i:(i + 1)]
        history = model.predict(x = [test_x_input, actual_decoder], verbose = 1, batch_size = 200)
        value = 0
        if history > 0.5:
            value = 1
        else:
            value = - 1
        partial_result_list[0][i + 2] = value
        print('prediction: ' + str(history) + "  ;  " + 'true_value: ' + str(testy[i]))


def modified_prediction(new_sequences_x, count, length_of_full_sequence, model = None):
    if model is None:
        model = load_model('/model_new_test_03', custom_objects = {'TransformerBlock': TransformerBlock,
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


def load_model_and_do_stuff(testx, testy, model):
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
    for i in range(5):
        x = []
        xx = 1
        col = []
        for j in testy[i]:
            x.append(xx)
            xx += 1
            if j > 0.5:
                col.append('red')
            else:
                col.append('blue')

        # print(prediction)
        plt.figure(figsize = (10, 10))
        plt.scatter(x, prediction[i], c = col)
        # plt.plot(testy[i], 'o',alpha=0.3, color='red')
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
        for j in range(len(trainx2[0]) - 3):

            if i[j + 3] == -1:
                new_trainy_parts.append(0)
            elif i[j + 3] == 1:
                new_trainy_parts.append(i[j + 3])
            else:
                new_trainy_parts.append(2)
            new_trainx2_parts.append(trainx2[counter])

            new_y = []
            for y, char in enumerate(i[:(j + 2)]):
                if char == -1:
                    new_y.append(0.5)
                elif char == 1:
                    new_y.append(1.)
                else:
                    new_y.append(0.)

            for y in range(len((trainx2[0]) - 3) - len(new_y)):
                new_y.append(0)

            new_trainy2_parts.append(new_y)

        # import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

        new_trainx2.append([new_trainx2_parts])
        new_trainy2.append([new_trainy2_parts])
        new_trainy.append([new_trainy_parts])

    return new_trainx2, new_trainy2, new_trainy


"""
try:
  TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_WORKER)  # TPU detection
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
  raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)
"""
#create_ai('/content/drive/MyDrive/ifp/Dataset-without-1550.xlsx')
use_model_and_predict()