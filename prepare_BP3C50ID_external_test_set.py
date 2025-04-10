"""
This Skripts sole purpose is transforming the raw BP3C50ID_external_test_set or training_set in the proper form for AI usage.
Embed the sequence and create the epitope sequence i.e. 0001110011010010100... where 0 is non-epitope and 1 is epitope residue.

Author: Florian Zwicker
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
    filepath = 'data/BP3C50ID/BP3C50ID_external_test_set_original.fasta'


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

    with open('AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    with open("AI/tokenizer_new.pkl", "wb") as handle:
        pickle.dump(encoder, handle)




    embedded_seq = encoder.texts_to_sequences(seq)


    padded_seq = pad_sequences(embedded_seq, maxlen = 235, padding = 'post', value = 0)

    df["Sequenz"] = list(padded_seq)

    return df



if __name__=="__main__":
    df = read_file()
    df = create_epitope_sequence(df)
    df = embed_sequence(df)
    df.to_csv("data/BP3C50ID/BP3C50ID_embedded_and_epitopes.csv", index=False)