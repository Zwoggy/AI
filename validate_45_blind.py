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
    print(df)

    fixed_length = 235
    # Feste Länge

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
    ### hier if länge >235
    sequences, epitope_list = keep_sequences_up_to_a_length_of_235(encoded_sequences, epitope_list, sequence_list)

    # Debugging step to check lengths
    for idx, epitope in enumerate(epitope_list):
        print(f"Length of epitope at index {idx}: {len(epitope)}")



    # Alle Sequenzen auf Länge 235 polstern (Padding mit 0)
    padded_sequences = sequence.pad_sequences(sequences, maxlen=fixed_length,
                                              padding='post', value=0)


    # Falls Sie trotzdem sequence.pad_sequences möchten, ginge das so:
    padded_epitope_list = sequence.pad_sequences(epitope_list, maxlen=fixed_length,
                                                 padding='post', value=0)

    # Modell evaluieren – je nach Bedarf anpassen. Die Funktion evaluate_model
    # muss ggf. so geschrieben sein, dass sie die Listen verarbeiten kann.
    results = []
    for idx, (seq, epi) in enumerate(zip(padded_sequences, padded_epitope_list)):
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



def keep_sequences_up_to_a_length_of_235(sequences, epitope_list, sequence_list):
    """
    Beschränkt alle Sequenzen auf eine maximale Länge von 235 Zeichen.

    Sequenzen, die bereits 235 oder weniger Zeichen haben, werden unverändert übernommen.
    Für längere Sequenzen wird der Teil mit den meisten Epitopen ausgewählt und
    auf 235 Zeichen gekürzt.

    :param sequences: Liste von (Index, Sequenz)-Tupeln, wobei jede Sequenz eine Zeichenkette ist
    :param epitope_list: Liste von Epitopen, die den Sequenzen zugeordnet sind
    :return: Ein Tupel bestehend aus zwei Listen:
             - new_sequences_list: Liste der modifizierten Sequenzen (alle maximal 235 Zeichen lang)
             - new_epitope_list: Liste der entsprechend modifizierten Epitope
    """

    new_sequences_list: list = []
    new_epitope_list: list = []
    for i, sequence in enumerate(sequences):
        if len(sequence) <= 235: # ist eine Sequenz kleiner oder gleich der maximal gewollten Länge, dann wird diese so beibehalten um die maximale Menge an Informationen zu behalten
            new_sequences_list.append(sequence)
            new_epitope_list.append(epitope_list[i])
        else: # ist eine Sequenz länger, dann wird eine Subsequenz der Länge von 235 herausgeschnitten
            new_sequence, new_epitope = prepare_sequence_part_of_length_235_with_most_epitopes(sequence, epitope_list[i], sequence_list[i])
            new_sequences_list.append(new_sequence)
            new_epitope_list.append(new_epitope)

    return new_sequences_list, new_epitope_list


def prepare_sequence_part_of_length_235_with_most_epitopes(sequence, epitope, sequence_list):
    """
        Extrahiert einen Teilabschnitt der Sequenz mit einer maximalen Länge von 235 Zeichen,
        der die meisten Epitope enthält.

        Die Funktion findet den Startpunkt des ersten Epitops (erste '1' im Epitope-Array)
        und wählt basierend darauf einen Teilabschnitt der Sequenz aus. Die Auswahl erfolgt
        nach unterschiedlichen Strategien, abhängig davon, wie viele Epitope nach dem Startpunkt folgen.

        :param sequence: Die vollständige Sequenz als Zeichenkette oder Liste
        :param epitope: Eine binäre Liste, die die Position der Epitope markiert (1 für Epitop, 0 sonst)
        :return: Ein Tupel bestehend aus:
                 - partial_sequence: Eine Liste mit dem ausgewählten Sequenzteilstück
                 - partial_epitope: Eine Liste mit dem entsprechenden Epitopteilstück
        """

    partial_sequence = []
    partial_epitope = []
    print("epitope: ", type(epitope), epitope)
    print("epitope_count: ", epitope.count("1"))
    try:
        epitope_start = epitope.index("1")
        if (len(epitope) - epitope_start) < 235:  # wenn die Subsequenz kürzer als 235 wäre, wird der Start nach Vorne geschoben bis die Länge 235 ist
            partial_sequence.append(sequence[(epitope_start - (len(epitope) - epitope_start)):len(epitope)])
            partial_epitope.append(epitope[(epitope_start - (len(epitope) - epitope_start)):len(epitope)])
        else:  # ansonsten wird genau dieser 235 lange Abschnitt der Sequenz als Subsequenz gespeichert
            partial_sequence.append(sequence[epitope_start:epitope_start + 235])
            partial_epitope.append(epitope[epitope_start:epitope_start + 235])

        return partial_sequence, partial_epitope
    except:
        print("In der folgenden Sequenz sind die Epitope nicht korrekt angegeben: ", sequence, sequence_list)
        return sequence[:235], epitope[:235]



