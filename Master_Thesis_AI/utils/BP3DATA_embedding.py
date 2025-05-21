"""
This script is used to read and return the BP3_training_set.fasta
"""


import keras
import types
import sys
import pickle
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from keras_preprocessing.text import Tokenizer
# Alias für das alte Modul setzen
keras.preprocessing = types.ModuleType("preprocessing")
keras.preprocessing.text = types.ModuleType("text")
keras.preprocessing.text.Tokenizer = Tokenizer
sys.modules["keras.preprocessing.text"] = keras.preprocessing.text





def read_file():
    """reads the file and
    :returns df of ID, Sequences """
    filepath = 'C:/Users/fkori/PycharmProjects/AI/data/BP3_Data/BP3_training_set.fasta'


    with open(filepath, 'r') as f:
        lines = f.readlines()

    ids = []
    seqs = []
    current_seq = ''

    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if current_seq:
                seqs.append(current_seq)
                current_seq = ''
            ids.append(line)
        else:
            current_seq += line

    # Letzte Sequenz anhängen
    if current_seq:
        seqs.append(current_seq)

    # Sicherheitscheck
    if len(ids) != len(seqs):
        raise ValueError(f"Mismatch: {len(ids)} IDs vs {len(seqs)} Sequenzen")

    # Erstelle DataFrame
    df = pd.DataFrame({'ID': ids, 'Sequenz': seqs})
    return df



def create_epitope_sequence(df):
    sequences = df["Sequenz"]

    epitope_sequences = []
    for sequence in sequences:
        epitope = ""
        for char in sequence:
            if char.isupper():
                epitope += "1"
            else:
                epitope += "0"
        epitope_sequences.append(epitope)
    df['Epitop'] = epitope_sequences


    return df


def embed_sequence(df):
    seq = df["Sequenz"]
    epi = df["Epitop"]
    epi = [list(map(int, list(e))) for e in epi] # transform single string into list of integers

    longest_seq = seq.loc[seq.str.len().idxmax()]
    maxlen = len(longest_seq)

    with open('C:/Users/fkori/PycharmProjects/AI/AI/tokenizer_new.pkl', 'rb') as handle:
        encoder = pickle.load(handle)


    embedded_seq = encoder.texts_to_sequences(seq)


    padded_seq = pad_sequences(embedded_seq, maxlen = maxlen, padding = 'post', value = 0)
    padded_epi = pad_sequences(epi, maxlen = maxlen, padding = 'post', value = -1)

    df["Sequenz"] = list(padded_seq)
    df["Epitop"] = list(padded_epi)

    return df



if __name__=="__main__":
    df = read_file()
    df = create_epitope_sequence(df)
    df = embed_sequence(df)
    df.to_csv('C:/Users/fkori/PycharmProjects/AI/data/BP3_Data/BP3_training_set_transformed.csv', index=False)