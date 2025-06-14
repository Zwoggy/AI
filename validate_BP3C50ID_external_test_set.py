import pickle
import pandas as pd
from keras_preprocessing.sequence import pad_sequences
from tf_keras.preprocessing import text

from ai_functionality_old import load_model_and_tokenizer, modify_with_context, evaluate_model




def validate_on_BP3C59ID_external_test_set(old_model=False, model=None):
    import pandas as pd
    import numpy as np
    from keras_preprocessing import text, sequence


    # Laden des Modells und Tokenizers (eigene Funktion anpassen)
    if old_model:
        model, _ = load_model_and_tokenizer()
    else:
        model = model

    # CSV-Datei einlesen
    df = pd.read_csv('./data/BP3C50ID/BP3C50ID_embedded_and_epitopes.csv')
    print(df)

    fixed_length = 235
    # Feste Länge

    sequence_list = []
    epitope_list = []

    # Durchlaufen der Zeilen im DataFrame und epitope_embed entsprechend befüllen
    encoded_sequences = df["Sequenz"]
    epitope_list = df["Epitop"]
    ### hier if länge >235
    sequences, epitope_list = keep_sequences_up_to_a_length_of_235(encoded_sequences, epitope_list)
    print("original_epitope_list: ", epitope_list)

    with open('./AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    sequences = [string_to_int_list(seq_str) for seq_str in sequences]

    # Alle Sequenzen auf Länge 235 polstern (Padding mit 0)
    padded_sequences = sequence.pad_sequences(sequences, maxlen=fixed_length,
                                              padding='post', value=0)

    epitope_list = [[int(char) for char in epitope] for epitope in epitope_list] # Für Padding vorbereiten, erwartet eine Liste von Integern
    print("epitope_list: ",epitope_list)
    #Alle Eitope auf die Länge 235 polstern (Padding mit 0)
    padded_epitope_list = sequence.pad_sequences(epitope_list, maxlen=fixed_length,
                                                 padding='post', value=0)
    print("padded_epitopes: ", padded_epitope_list)

    # Modell evaluieren – je nach Bedarf anpassen. Die Funktion evaluate_model
    # muss ggf. so geschrieben sein, dass sie die Listen verarbeiten kann.
    results = []
    for idx, (seq, epi) in enumerate(zip(padded_sequences, padded_epitope_list)):

        print("epi", epi)
        # Auswertung
        # PDB-ID oder ähnliches aus df entnehmen
        pdb_id = df['ID'].iloc[idx]

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
    results_df.to_csv('./data/evaluation_results_BP3C50ID_embedded_and_epitopes_14_06_2025.csv', index=False)
    print("Evaluation abgeschlossen und in 'evaluation_results_BP3C50ID_embedded_and_epitopes_14_06_2025.csv' gespeichert.")



def keep_sequences_up_to_a_length_of_235(sequences, epitope_list):
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
    print(epitope_list)
    new_sequences_list: list = []
    new_epitope_list: list = []
    for i, sequence in enumerate(sequences):
        if len(sequence) <= 235: # ist eine Sequenz kleiner oder gleich der maximal gewollten Länge, dann wird diese so beibehalten um die maximale Menge an Informationen zu behalten
            new_sequences_list.append(sequence)
            new_epitope_list.append(epitope_list[i])
        else: # ist eine Sequenz länger, dann wird eine Subsequenz der Länge von 235 herausgeschnitten
            print(i)
            new_sequence, new_epitope = prepare_sequence_part_of_length_235_with_most_epitopes(sequence, epitope_list[i])
            new_sequences_list.append(new_sequence)
            new_epitope_list.append(new_epitope)

    return new_sequences_list, new_epitope_list


def prepare_sequence_part_of_length_235_with_most_epitopes(sequence, epitope):
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
        print("epitope_start: ", epitope_start)

        if (len(epitope) - epitope_start) < 235:
            # Berechne, wie viele Zeichen vor der ersten "1" notwendig sind, damit die Subsequenz insgesamt 235 Zeichen lang ist
            start_index = max(0, epitope_start - (235 - (len(epitope) - epitope_start)))
            # Extrahiere die Subsequenz so, dass sie 235 Zeichen umfasst
            print("Epitoplänge der Berechnung IF: ", len(epitope[epitope_start - start_index:]))
            partial_sequence = sequence[start_index:start_index + 235]
            partial_epitope = epitope[start_index:start_index + 235]
        else:
            # Wenn die Distanz groß genug ist, einfach die 235 Zeichen ab der ersten "1"
            partial_sequence = sequence[epitope_start: epitope_start + 235]
            print("Epitoplänge der Berechnung ELSE: ", len(epitope[epitope_start: epitope_start + 235]))

            partial_epitope = epitope[epitope_start: epitope_start + 235]

        return partial_sequence, partial_epitope
    except:
        print("In der folgenden Sequenz sind die Epitope nicht korrekt angegeben: ", sequence)
        return sequence[:235], epitope[:235]



def string_to_int_list(s):
    return [int(x) for x in s.strip().replace('[', '').replace(']', '').split()]