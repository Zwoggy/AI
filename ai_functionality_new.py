# from tensorflow.core.framework.op_def_pb2 import tensorflow_dot_core_dot_framework_dot_full__type__pb2
# from pandas.io.pytables import AppendableMultiSeriesTable

# from typing_extensions import ParamSpecArgs
# from tensorflow.python.eager.context import ContextSwitch
###########################START##############################
import os
import math


import tensorflow as tf
import tensorflow
from tf_keras.src.utils import pad_sequences

from ai_functionality_old import get_weighted_loss

print("Tensorflow version " + tf.__version__)
#from tensorflow.python import keras


print('Start 1')

import sys
from sklearn.metrics import recall_score, precision_score, f1_score
import pickle
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from tensorflow.keras import layers

from tf_keras.models import load_model ### Using Keras 2

import random
from scipy.stats import gmean
#import keras_nlp
#from keras_nlp.layers import TokenAndPositionEmbedding as TAPE
#from keras_nlp.layers import TransformerEncoder as TE
import keras_preprocessing as kp
sys.modules['keras.preprocessing'] = kp
from keras_preprocessing import text
from keras_preprocessing.sequence import pad_sequences
# set seed to counter rng during training
### New imports for ESM-2
from transformers import EsmTokenizer
from tensorflow.keras import backend as K

random.seed(10)
tf.random.set_seed(10)

class LayerGroup(tf.keras.layers.Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layers:
            if isinstance(x, tuple):
                x = x[0]  # Nimm den ersten Wert aus dem Tuple
            x = layer(x, **kwargs)
        return x

# class TransformerBlock(tf.keras.Model):
class TransformerBlock(tensorflow.keras.layers.Layer):

    def __init__(self, embed_dim = 256, num_heads = 4, ff_dim = 32, rate = 0.3, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate


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
        self.ffn = tensorflow.keras.Sequential(
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
class TokenAndPositionEmbedding(tensorflow.keras.layers.Layer):

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

class TokenAndPositionEmbedding_for_ESM(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        """
        Klasse zur Kombination von Positions-Embeddings mit bereits existierenden Token-Embeddings.
        :param maxlen: Maximale Sequenzlänge.
        :param embed_dim: Dimensionalität der Embeddings.
        """
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        # Nur Positions-Embeddings, da die Token-Embeddings bereits durch das ESM-Modell bereitgestellt werden
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        """
        Kombiniert die Eingabe-Embeddings mit Positionsinformationen.
        :param x: Tensor der Form (batch_size, seq_len, embed_dim).
        :return: Tensor der Form (batch_size, seq_len, embed_dim) mit hinzugefügten Positionsinformationen.
        """
        batch_size = tf.shape(x)[0]  # Batch-Größe
        seq_len = tf.shape(x)[1]    # Dynamische Sequenzlänge

        # Positions-Indices erstellen: [0, 1, ..., seq_len-1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = self.pos_emb(positions)  # Embedding für die Positionen anwenden
        positions = tf.expand_dims(positions, axis=0)  # Batch-Dimension hinzufügen
        positions = tf.tile(positions, [batch_size, 1, 1])  # Für alle Batch-Elemente duplizieren

        return x + positions  # Positions-Embeddings zu den Eingaben addieren

    def compute_mask(self, inputs, mask=None):
        """
        Gibt die Maske unverändert weiter. Nützlich, wenn Maskierungen bei der Verarbeitung benötigt werden.
        """
        return mask

    def get_config(self):
        """
        Speichert die Konfigurationsparameter der Schicht, um sie später rekonstruieren zu können.
        """
        config = super().get_config()
        config.update({
            'maxlen': self.maxlen,
            'embed_dim': self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Rekonstruiert die Schicht aus der gespeicherten Konfiguration.
        """
        return cls(**config)



# class TokenAndPositionEmbedding2(tf.keras.Model):
class TokenAndPositionEmbedding2(tensorflow.keras.layers.Layer):

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
class TransformerDecoder(tensorflow.keras.layers.Layer):
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
        self.dense_proj = tensorflow.keras.Sequential(
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


class TransformerDecoderTwo(tensorflow.keras.layers.Layer):
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
        self.dense_proj = tensorflow.keras.Sequential(
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

    new_decoder_x = tensorflow.keras.preprocessing.sequence.pad_sequences(new_decoder_x, maxlen = maxlen, padding = 'post',
                                                               value = 0)

    return new_encoder_x, new_decoder_x, new_y




def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    #print("y_true: ",y_true)
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', classes = np.unique([0, 1]), y = np.ravel(y_true, order = 'C'))

    #print("WEIGHTS")
    #print(weights)
    for i in range(len(weights)):
        weights[i][1] = weights[i][1] / 2.2 # used to be 2.2

    #print("New WEIGHTS")
    #print(weights)

    return weights





import tensorflow as tf



def save_ai(model, path="./AI/EMS2_AI/AI"):
    model.save_weights(path + '_weights')
    model.save(path + '_model')


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
        tensorflow.keras.backend.clear_session()
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
        tensorflow.keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
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
    from transformers import AutoTokenizer, TFEsmForTokenClassification, TFEsmModel
    esm_model = TFEsmForTokenClassification.from_pretrained("facebook/esm2_t33_650M_UR50D")

    new = True
    """Enter a sequence to use for prediction and generate the heatmap output.
    All path need to be changed to wherever the files are stored on your computer."""
    sequence = "tpenitdlcaeyhntqihtlnnkifsyteslagkremaiitfkdgatfevevpgsehidsekkaiermkdtlriaylteakveklcvwnnktphaiaaisman"  # Hier die Sequenz eingeben#
    tensorflow.keras.backend.clear_session()
    """change the following path to the final_AI folder path"""
    model = load_model('./AI/EMS2_AI_model',
                       custom_objects = {'TransformerBlock': TransformerBlock,
                                         'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                         'TransformerDecoder': TransformerDecoder, "weighted_loss": get_weighted_loss,
                                         'esm_model': esm_model(235,)},
                       compile = False
                       )
    """change the following path to path/final_AI_weights """
    model.load_weights('./AI/EMS2_AI/AI_weights')
    model.compile()
    tensorflow.keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
                              to_file = './testpicture.png', show_layer_activations = True)
    print(model.summary(expand_nested = True))
    sequence_list = split_string(sequence)

    """change the following path accordingly"""

    with open('./AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)


    pre_embedded_docs = encoder.texts_to_sequences(sequence_list)
    embedded_docs = pad_sequences(pre_embedded_docs, maxlen = 235, padding = 'post',
                                                               value = 0)
    if new:
        embedded_docs = new_embedding(embedded_docs, encoder)

    predictions = model.predict(embedded_docs)


    x = 1
    pred_list = []
    sequence_list_for_further_stuff = []
    for i, (pred, seq) in enumerate(zip(predictions, sequence)):
        for (j, seq2) in zip(pred, sequence):
            pred_list.append(j)
            sequence_list_for_further_stuff.append(seq2)
            print(str(x) + ": " + str(j) + " - " + str(sequence[x - 1]))
            x += 1

    create_better_heatmap(pred_list, sequence, sequence_list_for_further_stuff)


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

def create_heatmap(data, sequence):
    """Input: predictions from the model
    Output: Heatmaps according to the predictions for the whole sequence entered"""
    data = np.array(data[:len(sequence)], dtype = np.float32)
    # data = np.reshape(data, (data.shape[1], data.shape[0]))

    data_list, sequence_list = create_blocks(data, sequence)
    print(data_list, sequence_list)

    for i, (pred, seq) in enumerate(zip(data_list, sequence_list)):
        """change the path to a folder to save the pictuers in"""
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
        block1 = np.array(list1[start:end])
        block2 = np.array(list2[start:end])
        blocks1.append(block1)
        blocks2.append(block2)

    return np.array(blocks1), np.array(blocks2)

###################################################################################
def transformer_prediction_loop(testx, test_decoder_x, testy):
    tensorflow.keras.backend.clear_session()
    model = load_model('/content/drive/MyDrive/ifp/ki_für_tests_without_decoder_MODEL',
                       custom_objects = {'TransformerBlock': TransformerBlock,
                                         'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                         'TokenAndPositionEmbedding2': TokenAndPositionEmbedding2,
                                         'TransformerDecoder': TransformerDecoder},
                       compile = False
                       )
    model.load_weights('/content/drive/MyDrive/ifp/ki_für_tests_without_decoder_MODEL_weights')
    model.compile()
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

"""these are the functions that are being executed when running the python script.
Use create_ai() to train a new model.
Use use_model_and_predict() to use the model for prediction.
"""
#path = 'C:/Users/fkori/PycharmProjects/AI/Dataset.xlsx'











def use_model_and_predict_45_blind(sequence, model, encoder):
    """Verwendet ein Modell zur Vorhersage einer Sequenz und gibt die Wahrscheinlichkeiten zurück."""
    # Sequence in eine Liste aufsplitten
    sequence_list = split_string(sequence)

    # Encode die Sequenz mit dem gespeicherten Tokenizer
    pre_embedded_docs = encoder.texts_to_sequences(sequence_list)
    embedded_docs = pad_sequences(pre_embedded_docs, maxlen=235, padding='post',
                                                                  value=0)

    # Vorhersagen
    predictions = model.predict(embedded_docs)

    # Reduziere die Vorhersagen auf 1D
    return predictions.flatten()[:235]


def load_model_and_tokenizer():
    """Lädt das Modell und den Tokenizer."""
    # Lade das Modell
    model = load_model('G:/Users/tinys/PycharmProjects/teststuff/AI/final_AI',
                       custom_objects={'TransformerBlock': TransformerBlock,
                                       'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                       'TransformerDecoder': TransformerDecoder,
                                       "weighted_loss": get_weighted_loss},
                       compile=False)
    # Lade die Gewichte
    model.load_weights('G:/Users/tinys/PycharmProjects/teststuff/AI/final_AI_weights')
    model.compile()
    # Lade den Tokenizer
    with open('G:/Users/tinys/PycharmProjects/teststuff/AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    return model, encoder


def evaluate_model(model, encoder, sequence, true_binary_epitope):
    """Vergleicht die Vorhersagen des Modells mit dem tatsächlichen Epitop-Binärstring."""
    predictions = use_model_and_predict_45_blind(sequence, model, encoder)

    # Da das Modell Wahrscheinlichkeiten ausgibt, runde auf 0 oder 1
    predicted_binary = np.where(predictions >= 0.5, 1, 0)

    # Berechne die Metriken
    #auc = roc_auc_score(true_binary_epitope, predictions)
    recall = recall_score(true_binary_epitope, predicted_binary)
    precision = precision_score(true_binary_epitope, predicted_binary)
    f1 = f1_score(true_binary_epitope, predicted_binary)

    return recall, precision, f1 #,auc






