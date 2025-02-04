import pickle

from ai_functionality_old import load_model_and_tokenizer, modify_with_context, evaluate_model




def validate_on_45_blind():
    import pandas as pd
    import numpy as np
    from keras_preprocessing import text, sequence


    # Laden des Modells und Tokenizers (eigene Funktion anpassen)
    model, _ = load_model_and_tokenizer()

    # CSV-Datei einlesen
    df = pd.read_csv('./data/epitope3d_dataset_45_Blind_Test_manual_with_epitopes2.csv')

    # Feste Länge
    fixed_length = 235

    sequence_list = []
    epitope_list = []

    # Durchlaufen der Zeilen im DataFrame und epitope_embed entsprechend befüllen
    for idx, row in df.iterrows():
        full_sequence = str(row['Sequence'])

        # Epitope-Array mit -1 initialisieren
        epitope_embed = [-1] * fixed_length

        # Falls Epitope-Informationen 0/1-codiert sind, hier aus der Spalte entnehmen und eintragen
        # Beispiel: 'Epitope Sequence' enthält ein String-Array aus 0ern/1ern oder ähnlichem
        # Passen Sie dies an Ihr tatsächliches Format an.
        raw_epitope_info = str(row['Epitope Sequence']).replace(" ", "")

        # Beispiel: wenn raw_epitope_info eine Liste von Ziffern "0" oder "1" ist
        # und deren Länge der tatsächlichen Sequenz entspricht
        for i, c in enumerate(raw_epitope_info):
            if i < fixed_length:
                if c == '1':
                    epitope_embed[i] = 1

        # Optional: Mindestanzahl an Epitope überprüfen
        if epitope_embed.count(1) > 4:
            # Sequenz abspeichern (wird später tokenisiert)
            sequence_list.append(full_sequence)
            # Liste der Epitope
            epitope_list.append(epitope_embed)

    # Tokenizer laden (oder neu anlegen, je nach Bedarf)
    with open('./AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    # Die erfassten Sequenzen mithilfe des Tokenizers in Zahlen umwandeln
    encoded_sequences = encoder.texts_to_sequences(sequence_list)

    # Alle Sequenzen auf Länge 235 polstern (Padding mit 0)
    padded_sequences = sequence.pad_sequences(encoded_sequences, maxlen=fixed_length,
                                              padding='post', value=0)

    # Auch epitope_list polstern (falls nötig) – in diesem Beispiel sind sie schon fix 235
    # Falls Sie trotzdem sequence.pad_sequences möchten, ginge das so:
    padded_epitope_list = sequence.pad_sequences(epitope_list, maxlen=fixed_length,
                                                 padding='post', value=0)

    # Modell evaluieren – je nach Bedarf anpassen. Die Funktion evaluate_model
    # muss ggf. so geschrieben sein, dass sie die Listen verarbeiten kann.
    results = []
    for idx, (seq, epi) in enumerate(zip(padded_sequences, padded_epitope_list)):
        # Auswertung
        # PDB-ID oder ähnliches aus df entnehmen
        pdb_id = df['PDB ID'].iloc[idx]

        # Beispielhafter Aufruf
        recall, precision, f1 = evaluate_model(model, encoder, [seq], epi)

        results.append({
            'PDB ID': pdb_id,
            'Recall': recall,
            'Precision': precision,
            'F1-Score': f1
        })

    # Ergebnisse in CSV speichern
    results_df = pd.DataFrame(results)
    results_df.to_csv('evaluation_results.csv', index=False)
    print("Evaluation abgeschlossen und in 'evaluation_results.csv' gespeichert.")




