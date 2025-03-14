import pickle
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from tf_keras.preprocessing import text

from ai_functionality_old import load_model_and_tokenizer, modify_with_context, evaluate_model




def validate_on_45_blind():
    import pandas as pd
    import numpy as np
    from keras_preprocessing import text, sequence


    # Laden des Modells und Tokenizers (eigene Funktion anpassen)
    model, _ = load_model_and_tokenizer()

    # CSV-Datei einlesen
    df = pd.read_csv('./data/final_blind_test_set.csv')

    # Feste Länge
    fixed_length = 235

    sequence_list = []
    epitope_list = []

    # Durchlaufen der Zeilen im DataFrame und epitope_embed entsprechend befüllen
    for idx, row in df.iterrows():
        full_sequence = str(row['Sequence'])

        # Epitope-Array mit -1 initialisieren

        # Falls Epitope-Informationen 0/1-codiert sind, hier aus der Spalte entnehmen und eintragen
        # Beispiel: 'Epitope Sequence' enthält ein String-Array aus 0ern/1ern oder ähnlichem
        # Passen Sie dies an Ihr tatsächliches Format an.
        raw_epitope_info = str(row['Epitope Sequence']).replace(" ", "")


        # Sequenz abspeichern (wird später tokenisiert)
        sequence_list.append(full_sequence)
        # Liste der Epitope
        epitope_list.append(raw_epitope_info)

    # Tokenizer laden (oder neu anlegen, je nach Bedarf)
    with open('./AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    # Die erfassten Sequenzen mithilfe des Tokenizers in Zahlen umwandeln
    encoded_sequences = encoder.texts_to_sequences(sequence_list)
    epitope_list, antigen_list, length_of_longest_context = modify_with_context(encoded_sequences, epitope_list,
                                                                                fixed_length)
    print(epitope_list)

    """
    # Alle Sequenzen auf Länge 235 polstern (Padding mit 0)
    padded_sequences = sequence.pad_sequences(encoded_sequences, maxlen=fixed_length,
                                              padding='post', value=0)


    # Falls Sie trotzdem sequence.pad_sequences möchten, ginge das so:
    padded_epitope_list = sequence.pad_sequences(epitope_list, maxlen=fixed_length,
                                                 padding='post', value=0)
    """
    print("Anzahl der Sequenzen: ",len(antigen_list))
    # Modell evaluieren – je nach Bedarf anpassen. Die Funktion evaluate_model
    # muss ggf. so geschrieben sein, dass sie die Listen verarbeiten kann.
    results = []
    for idx, (seq, epi) in enumerate(zip(antigen_list, epitope_list)):
        print("epi", epi)
        # Auswertung
        # PDB-ID oder ähnliches aus df entnehmen
        pdb_id = df['PDB ID'].iloc[idx]

        # Beispielhafter Aufruf
        recall, precision, f1, auc = evaluate_model(model, encoder, seq, epi)

        results.append({
            'PDB ID': pdb_id,
            'Recall': recall,
            'Precision': precision,
            'F1-Score': f1,
            'AUC': auc
        })

    # Ergebnisse in CSV speichern
    results_df = pd.DataFrame(results)
    results_df.to_csv('./data/evaluation_results_2.csv', index=False)
    print("Evaluation abgeschlossen und in 'evaluation_results_2.csv' gespeichert.")





