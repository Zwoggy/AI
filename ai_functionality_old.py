# from tensorflow.core.framework.op_def_pb2 import tensorflow_dot_core_dot_framework_dot_full__type__pb2
# from pandas.io.pytables import AppendableMultiSeriesTable

# from typing_extensions import ParamSpecArgs
# from tensorflow.python.eager.context import ContextSwitch
###########################START##############################
import os
import math


import tensorflow as tf
from tf_keras.src.utils import pad_sequences

from src.TokenAndPositionEmbedding import TokenAndPositionEmbedding
from src.TokenAndPositionEmbedding2 import TokenAndPositionEmbedding2
from src.TransformerBlock import TransformerBlock
from src.TransformerDecoder import TransformerDecoder

print("Tensorflow version " + tf.__version__)
#from tensorflow.python import keras


print('Start 1')

import sys
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import pickle
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt

from tf_keras.models import load_model ### Using Keras 2
import tf_keras ### Keras 2
from tensorflow.keras import backend as K
#from tf_keras import backend as K
import random
from scipy.stats import gmean
#import keras_nlp
#from keras_nlp.layers import TokenAndPositionEmbedding as TAPE
#from keras_nlp.layers import TransformerEncoder as TE
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import keras_preprocessing as kp
sys.modules['keras.preprocessing'] = kp
from keras_preprocessing import text
from keras_preprocessing.sequence import pad_sequences
# set seed to counter rng during training
### New imports for ESM-2
from transformers import EsmTokenizer
from tensorflow.keras import backend as K
import keras

random.seed(10)
tf.random.set_seed(10)


def extract_sequence_from_pdb_simple(pdb_file_path):
    """
    Extracts the amino acid sequence from a PDB file based on residue information.
    """
    three_to_one = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }

    sequence = []
    with open(pdb_file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM") and line[13:15] == "CA":  # Alpha carbons denote residues
                residue_name = line[17:20].strip()
                sequence.append(three_to_one.get(residue_name, 'X'))  # Use 'X' for unknown residues

    return ''.join(sequence)


def load_structure_data(pdb_dir, sequence_list):
    """
    Load structural data for each sequence from corresponding PDB files.

    The sequence from the sequence list must fully exist as a substring
    in the sequence extracted from the PDB file.

    Parameters:
    ----------
    pdb_dir : str
        Directory containing PDB files.
    sequence_list : list
        List of sequences for which structural data needs to be retrieved.

    Returns:
    ----------
    list
        List of structural data corresponding to the sequences, with None for sequences without matches.
    """
    used_pdb_files = set()  # To track used PDB files
    structure_data = []

    for sequence in sequence_list:
        matched_structure = None

        for pdb_file in os.listdir(pdb_dir):
            if pdb_file in used_pdb_files:  # Skip already used files
                continue

            pdb_path = os.path.join(pdb_dir, pdb_file)
            if pdb_file.endswith('.pdb'):
                pdb_sequence = extract_sequence_from_pdb_simple(pdb_path)
                if sequence in pdb_sequence:  # Check if the sequence exists in the PDB sequence
                    matched_structure = pdb_file
                    used_pdb_files.add(pdb_file)  # Mark this file as used
                    break

        structure_data.append(matched_structure if matched_structure else None)
    none_count = structure_data.count(None)
    print(f"Anzahl der None-Einträge: {none_count}")
    pdb_count = sum(1 for item in structure_data if item and item.endswith('.pdb'))
    print(f"Anzahl der .pdb-Dateien: {pdb_count}")
    total_length = len(structure_data)
    print(f"Gesamtlänge der Liste: {total_length}")
    missing_files = total_length - (none_count + pdb_count)
    print(f"Fehlende Einträge: {missing_files}")

    return structure_data


def read_data(filepath):
    df = pd.read_excel(filepath, skiprows = [-1])

    sequence_as_aminoacids_list: list = []

    first_col: str = 'Epitope'
    epitope_embed_list: list = []
    accession_id_list: list = []
    print("Hier die Anzahl aller einzigartigen Accessions",df["Accession"].nunique())
    accession_id = df["Accession"].tolist()
    print(accession_id)

    for i, sequence in enumerate(df['Sequence']):
        """
        if pd.isna(accession_id[i]): # use to eliminate sequences without ID and thus without structure.
            continue
            """

        column = 3

        char_list: list = []

        sequence_as_sentence: str = ""

        epitope_embed: list = []

        loc_column = first_col

        for char in str(sequence):
            char_list.append(char)
            ### + " "
            sequence_as_sentence += char

            epitope_embed.append(0) # IMPORTANT! Used to be -1 as non-epitopes where marked -1

        while column < 234:

            epitope_encoded: list = str(df.loc[i, loc_column]).replace(" ", "").split(",")

            for yeetitope in epitope_encoded:

                if yeetitope != "nan":
                    epitope_embed[int(yeetitope[1:]) - 1] = 1

            loc_column = 'Unnamed: ' + str(column)
            column += 1

        # if (epitope_embed.count(1) > 4) and epitope_embed.count(1) < 18 :
        if epitope_embed.count(1) > 7: # used to be 4
            epitope_embed_list.append(epitope_embed)
            sequence_as_aminoacids_list.append(sequence_as_sentence)
            accession_id_list.append(accession_id[i])

    return sequence_as_aminoacids_list, epitope_embed_list, accession_id_list

def embedding(filepath, old=False):
    """
    Embeds sequences and epitope data from a given filepath.

    Parameters:
    ----------
    filepath : str
        Path to the file containing the sequence and epitope data.
    old : bool, optional
        Flag to determine the usage of the old embedding method (default is False).

    Returns:
    ----------
    tuple or None
        returns a tuple of:
        - embedded_docs: numpy.ndarray
            Embedded sequences of the same length.
        - epitope_embed_list: numpy.ndarray
            Padded sequences of epitope embeddings.
        - voc_size: int
            The vocabulary size used for embedding.
        - length_of_longest_sequence: int
            The length of the longest sequence used in padding.
        - encoder: keras.preprocessing.text.Tokenizer
            The tokenizer used for embedding the sequences.
    """

    sequence_list, epitope_embed_list, accession_id = read_data(filepath)

    voc_size = 100

    #length_of_longest_sequence = int(len(max(sequence_list, key = len)) / 10)
    #length_of_longest_sequence = int(len(max(sequence_list, key = len)))
    length_of_longest_sequence = 235

    epitope_embed_list = pad_sequences(epitope_embed_list, maxlen=length_of_longest_sequence,
                                       padding='post', value=-1) # IMPORTANT used to be 0 as padding was 0
    encoder = text.Tokenizer(num_words = 30, char_level = True,  oov_token="X")
    """
    with open('./AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)
    """
    """Usage for the old AI"""
    # loading


    encoder.fit_on_texts(sequence_list)
    pre_embedded_docs = encoder.texts_to_sequences(sequence_list)
    # saving

    #with open('/content/drive/MyDrive/ifp/tokenizer.pickle', 'wb') as handle:
    #  pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #print(encoder.word_index)

    embedded_docs = pad_sequences(pre_embedded_docs, maxlen = length_of_longest_sequence,
                                                            padding = 'post', value = 0)

    one_hot_matrix = encoder.texts_to_matrix(sequence_list, mode='binary')
    one_hot_embedded_docs = pad_sequences(one_hot_matrix, maxlen=length_of_longest_sequence, padding='post', value=0)

    # embedded_docs = np.array(embedded_docs)

    max_len_antigen: int = len(max(epitope_embed_list, key = len))

    # embedded_docs = np.array(embedded_docs)


    return embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder, accession_id


def embedding_incl_structure(filepath, pdb_dir, old=False):
    """
    Embeds sequences and epitope data from a given filepath.

    Parameters:
    ----------
    filepath : str
        Path to the file containing the sequence and epitope data.
    old : bool, optional
        Flag to determine the usage of the old embedding method (default is False).

    Returns:
    ----------
    tuple or None
        returns a tuple of:
        - embedded_docs: numpy.ndarray
            Embedded sequences of the same length.
        - epitope_embed_list: numpy.ndarray
            Padded sequences of epitope embeddings.
        - voc_size: int
            The vocabulary size used for embedding.
        - length_of_longest_sequence: int
            The length of the longest sequence used in padding.
        - encoder: keras.preprocessing.text.Tokenizer
            The tokenizer used for embedding the sequences.
    """

    sequence_list, epitope_embed_list = read_data(filepath)

    voc_size = 100

    length_of_longest_sequence = int(len(max(sequence_list, key = len)) / 2)

    epitope_embed_list = pad_sequences(epitope_embed_list, maxlen=length_of_longest_sequence,
                                       padding='post', value=0)
    encoder = text.Tokenizer(num_words = 1000, char_level = True,  oov_token="X")
    """
    with open('./AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)
    """
    """Usage for the old AI"""
    # loading


    encoder.fit_on_texts(sequence_list)
    pre_embedded_docs = encoder.texts_to_sequences(sequence_list)
    # saving

    #with open('/content/drive/MyDrive/ifp/tokenizer.pickle', 'wb') as handle:
    #  pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #print(encoder.word_index)

    embedded_docs = pad_sequences(pre_embedded_docs, maxlen = length_of_longest_sequence,
                                                            padding = 'post', value = 0)


    # embedded_docs = np.array(embedded_docs)

    max_len_antigen: int = len(max(epitope_embed_list, key = len))

    # embedded_docs = np.array(embedded_docs)
    structure_data = load_structure_data(pdb_dir, sequence_list)

    return embedded_docs, epitope_embed_list, voc_size, length_of_longest_sequence, encoder, structure_data



def new_embedding(antigen_list, encoder):
    """
    Reverse the previous embedding and perform new embeddings using ESM-2.

    Parameters:
    ----------
    antigen_list : numpy.ndarray
        List of antigen sequences (as numerical sequences).
    epitope_list : numpy.ndarray
        List of epitope sequences (as numerical sequences).
    encoder : keras.preprocessing.text.Tokenizer
        The tokenizer used for embedding the sequences.
    length_of_longest_sequence : int
        Maximum length for padding.

    Returns:
    ----------
    tuple
        New embedded sequences and epitopes.
    """
    # Rekonstruktion der originalen Sequenzen mit dem Encoder
    #decoded_antigens = encoder.sequences_to_texts(antigen_list.tolist()) # alte Version / vielleicht ist die Liste notwendig
    decoded_antigens: list = encoder.sequences_to_texts(antigen_list)
    for i, decoded_antigen  in enumerate(decoded_antigens):
        decoded_antigens[i]: str = decoded_antigen.replace(" ", "")
    #print("decoded_antigens: ", decoded_antigens)

    # Lade ESM-Tokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
    tokenizer.pad_token_id = 0

    # Neue Embeddings für Antigen
    new_embedded_docs = []
    for doc in decoded_antigens:
        encoded_doc = tokenizer.encode_plus(
            doc,
            return_tensors='tf',  # TensorFlow verwenden
            padding='max_length',
            truncation=True,
            max_length=235,
            add_special_tokens=False,
            padding_side='right',
            return_attention_mask=True
        )
        # Entfernt Dimensionen mit Länge 1
        squeezed_input_ids = tf.squeeze(encoded_doc['input_ids']).numpy()  # tf.squeeze verwenden
        new_embedded_docs.append(squeezed_input_ids)


    new_embedded_docs = np.array(new_embedded_docs)
    return new_embedded_docs




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


# class TokenAndPositionEmbedding(tf.keras.Model):


# class TokenAndPositionEmbedding2(tf.keras.Model):


# class TransformerDecoder(tf.keras.Model):


"""
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

"""
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

    new_decoder_x = tf_keras.preprocessing.sequence.pad_sequences(new_decoder_x, maxlen = maxlen, padding = 'post',
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

                    number = random.randint(0, int(context / 2))

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
    new_epitope_list = tf_keras.preprocessing.sequence.pad_sequences(new_epitope_list, maxlen = length_of_longest_context,
                                                                  padding = 'post', value = -1)
    new_antigen_list = tf_keras.preprocessing.sequence.pad_sequences(new_antigen_list, maxlen = length_of_longest_context,
                                                                  padding = 'post', value = 0)

    return new_epitope_list, new_antigen_list, length_of_longest_context


def modify_with_max_epitope_density(epitope_list, antigen_list, window_size):
    """
    Finds the subsequence of a given window size that contains the most 1s in the epitope list.
    Returns the truncated antigen and epitope sequences, along with the new max length.

    """
    new_epitope_list = []
    new_antigen_list = []

    for epitope, antigen in zip(epitope_list, antigen_list):
        max_count = -1
        max_start = 0

        # Ensure length is valid
        if len(epitope) < window_size:
            pad_len = window_size - len(epitope)
            epitope += [0] * pad_len
            antigen += ['X'] * pad_len  # or any padding character you prefer

        # Slide the window
        for i in range(len(epitope) - window_size + 1):
            window = epitope[i:i + window_size]
            count = sum(1 for x in window if x == 1)
            if count > max_count:
                max_count = count
                max_start = i

        # Extract the window with the highest number of 1s
        selected_epitope = epitope[max_start:max_start + window_size]
        selected_antigen = antigen[max_start:max_start + window_size]

        new_epitope_list.append(selected_epitope)
        new_antigen_list.append(selected_antigen)

    # Padding to fixed length (optional)
    new_epitope_list = pad_sequences(new_epitope_list, maxlen=window_size, padding='post', value=-1)
    new_antigen_list = pad_sequences(new_antigen_list, maxlen=window_size, padding='post', value=0)

    return new_epitope_list, new_antigen_list, window_size


def modify_with_context_big_dataset(epitope_list, antigen_list, length_of_longest_sequence):
    """ The sequences are going to be cut into shorter pieces, where the first aminoacid being part of an epitope is marked as the start.
        A random number of non-epitope aminoacids will be added infront of the starting epitope.

        context: defines the length after which the sequence will be cut if no epitope was found.

        returns the new antigen(actually protein) and epitope list aswell as the new length of the longest sequence to which every new sequence is padded."""
    new_antigen_list: list = []
    new_epitope_list: list = []
    decoder_list: list = []
    context = 20
    for end in range (1,3):
        # to cut off 1 to 10 aminoacids from the end for more different sequences in the dataset
        for number in range(1, 10):
            # to cut off 1 to 10 aminoacids from the start for more different sequences in the dataset
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

                short_epitope = short_epitope[:-end]
                short_antigen = short_antigen[:-end]
                new_epitope_list.append(short_epitope)
                new_antigen_list.append(short_antigen)
                # Also append the reversed antigen and epitope
                new_epitope_list.append(short_epitope[::-1])
                new_antigen_list.append(short_antigen[::-1])
                # print(short_antigen)
                # print(short_epitope)

    length_of_longest_context = int(len(max(new_antigen_list, key = len)))
    # print(short_epitope)
    length_of_longest_context = 235
    new_epitope_list = tf_keras.preprocessing.sequence.pad_sequences(new_epitope_list, maxlen = length_of_longest_context,
                                                                  padding = 'post', value = 0)
    new_antigen_list = tf_keras.preprocessing.sequence.pad_sequences(new_antigen_list, maxlen = length_of_longest_context,
                                                                  padding = 'post', value = 0)

    return new_epitope_list, new_antigen_list, length_of_longest_context


def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])

    for i in range(number_dim):
        # Entferne Padding (-1)
        col = y_true[:, i]
        col = col[col != -1]  # Nur echte Labels (0 oder 1)

        if len(np.unique(col)) == 1:
            # Wenn nur eine Klasse vorkommt, setze Standardgewicht
            weights[i] = [1.0, 1.0]
        else:
            w = compute_class_weight('balanced', classes=np.array([0, 1]), y=col)
            weights[i] = w


    for i in range(len(weights)):
        weights[i][1] = weights[i][1] / 1.0  # optional Scaling; used to be 2.2; 0.75, 0.85

    return weights
"""
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
        weights[i][1] = weights[i][1] / 0.5 # used to be 2.2

    #print("New WEIGHTS")
    #print(weights)

    return weights
"""

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        # Wahrscheinlichkeiten für jede Klasse beschneiden (Num. Stabilität)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        # Berechne die Focal Loss
        cross_entropy_loss = -y_true * tf.math.log(y_pred)
        focal_loss_value = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy_loss
        return tf.reduce_sum(focal_loss_value, axis=1)

    return focal_loss_fixed


import tensorflow as tf

def stochastic_loss():
    def loss_function(y_true, y_pred, ignore_fraction=0.75):
        """
        Binary Crossentropy mit stochastischem Ignorieren von 0en, ohne explizite Masken.
        - ignore_fraction: Wahrscheinlichkeit, mit der 0en ignoriert werden.
        """
        # Zufällige Wahrscheinlichkeit für jedes Element
        random_values = tf.random.uniform(tf.shape(y_true))

        # Berechne BCE nur für relevante Werte
        loss = tf.where(
            (y_true == 0) & (random_values < ignore_fraction),  # Bedingung: Klasse 0 und ignorieren
            0.0,  # Setze den Loss auf 0
            y_true * tf.math.log(y_pred + tf.keras.backend.epsilon()) +
            (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())  # Standard-BCE
        )

        # Durchschnittlicher Verlust
        return -tf.reduce_mean(loss)

    return loss_function



def combined_focal_cross_entropy_loss(gamma=2.0, alpha=0.25, lambda_ce=0.5):
    def loss(y_true, y_pred):
        # Clip predictions for numerical stability
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        # Calculate Cross-Entropy Loss
        cross_entropy_loss = -y_true * tf.math.log(y_pred)
        cross_entropy_loss = tf.reduce_sum(cross_entropy_loss, axis=1)

        # Calculate Focal Loss
        focal_loss_value = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy_loss
        focal_loss_value = tf.reduce_sum(focal_loss_value, axis=1)

        # Combine Cross-Entropy and Focal Loss
        combined_loss = lambda_ce * cross_entropy_loss + (1 - lambda_ce) * focal_loss_value
        return combined_loss

    return loss

def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred),
            axis = -1)

    return weighted_loss




@keras.saving.register_keras_serializable()
def get_weighted_loss_masked_(weights):
    weights = tf.constant(weights, dtype=tf.float32)  # shape: (seq_len, 2)

    def weighted_loss_masked_(y_true, y_pred):
        # y_true: (batch, seq_len, 1), y_pred: (batch, seq_len, 1)
        y_true = tf.squeeze(y_true, axis=-1)  # → (batch, seq_len)
        y_pred = tf.squeeze(y_pred, axis=-1)

        # Maske: 1 für echte Werte, 0 für Padding
        mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)  # (batch, seq_len) # -1 padding token

        # Hole Gewichte: shape (seq_len, 2) → (1, seq_len, 2) für Broadcast
        w = tf.expand_dims(weights, axis=0)

        # Erzeuge Gewichtsmatrix: (batch, seq_len)
        weight_per_token = tf.where(tf.equal(y_true, 1), w[:, :, 1], w[:, :, 0])

        # Berechne Binary Crossentropy
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)  # (batch, seq_len)

        # Wende Gewichte und Maske an
        loss = bce * weight_per_token * mask

        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())

    return weighted_loss_masked_



def get_weighted_loss_masked(weights):
    weights = tf.constant(weights, dtype=tf.float32)  # shape: (seq_len, 2)

    def weighted_loss_masked(y_true, y_pred):
        # y_true: (batch, seq_len, 1), y_pred: (batch, seq_len, 1)
        y_true = tf.squeeze(y_true, axis=-1)  # → (batch, seq_len)
        y_pred = tf.squeeze(y_pred, axis=-1)

        # Maske: 1 für echte Werte, 0 für Padding
        mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)  # (batch, seq_len); padding wert = -1

        # Hole Gewichte: shape (seq_len, 2) → (1, seq_len, 2) für Broadcast
        w = tf.expand_dims(weights, axis=0)

        # Erzeuge Gewichtsmatrix: (batch, seq_len)
        weight_per_token = tf.where(tf.equal(y_true, 1), w[:, :, 1], w[:, :, 0])

        # Berechne Binary Crossentropy
        bce = tf_keras.backend.binary_crossentropy(y_true, y_pred)  # (batch, seq_len)

        # Wende Gewichte und Maske an
        loss = bce * weight_per_token * mask

        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + tf_keras.backend.epsilon())

    return weighted_loss_masked


def save_ai(model, path="./AI/EMS2_AI/AI", old=False):
    if old:
        model.save_weights(path + '_weights')
        model.save(path + '_model')
    else:
        model.save(path + "_model_keras_3")



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
        tf_keras.backend.clear_session()
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
        tf_keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
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
    tf_keras.backend.clear_session()
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
    tf_keras.utils.plot_model(model, expand_nested = True, show_shapes = True,
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
    tf_keras.backend.clear_session()
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
    #sequence_list = split_string(sequence)

    # Encode die Sequenz mit dem gespeicherten Tokenizer
    #pre_embedded_docs = encoder.texts_to_sequences(sequence_list)
    #embedded_docs = pad_sequences(pre_embedded_docs, maxlen=235, padding='post', value=0)

    seq_reshaped = np.expand_dims(sequence, axis=0)

    # Vorhersagen
    predictions = model.predict(seq_reshaped)
    # Reduziere die Vorhersagen auf 1D
    return predictions.flatten()[:235]


def load_model_and_tokenizer(model=None):
    """Lädt das Modell und den Tokenizer."""
    # Lade das Modell
    if model is None:
        model = load_model('./AI/final_AI',
                           custom_objects={'TransformerBlock': TransformerBlock,
                                           'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                           'TransformerDecoder': TransformerDecoder,
                                           "weighted_loss": get_weighted_loss},
                           compile=False)
        # Lade die Gewichte
        model.load_weights('./AI/final_AI_weights')
        model.compile()
    else:
        model=model

    # Lade den Tokenizer
    with open('./AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    return model, encoder


def evaluate_model(model, encoder, sequence, true_binary_epitope):
    """Vergleicht die Vorhersagen des Modells mit dem tatsächlichen Epitop-Binärstring."""

    predictions = use_model_and_predict_45_blind(sequence, model, encoder)

    # Da das Modell Wahrscheinlichkeiten ausgibt, runde auf 0 oder 1
    # print(true_binary_epitope)
    # print(predictions)
    predicted_binary = np.where(predictions >= 0.5, 1, 0)
    # Berechne die Metriken
    print( "test: ", true_binary_epitope, predicted_binary)
    auc = roc_auc_score(true_binary_epitope, predictions)
    recall = recall_score(true_binary_epitope, predicted_binary)
    precision = precision_score(true_binary_epitope, predicted_binary)
    f1 = f1_score(true_binary_epitope, predicted_binary)
    print(recall, precision, f1, auc)

    return recall, precision, f1 ,auc






