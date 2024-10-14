import pandas as pd
import requests

# CSV-Datei einlesen
file_path = 'epitope3d_dataset_45_Blind_Test.csv'
df = pd.read_csv(file_path)


# Funktion zum Abrufen der Sequenz von der PDB ID
def fetch_sequence_from_pdb(pdb_id):
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        sequence = ''.join(fasta_data.split('\n')[1:])
        return sequence
    else:
        print(f"Fehler beim Abrufen der Sequenz für {pdb_id}")
        return None


# Funktion zur Umwandlung der Epitope-Liste in eine Binärsequenz
def map_epitopes_to_binary(sequence_length, epitope_positions, start):
    binary_list = ['0'] * sequence_length
    for pos in epitope_positions:
        if start <= pos < start + sequence_length:
            binary_list[pos - start] = '1'
    return ''.join(binary_list)


# Parameter
subseq_length = 235

# Für jede Zeile im Datensatz
results = []

for idx, row in df.iterrows():
    pdb_id = row['PDB ID']
    epitope_list = row['Epitope List (residueid_residuename_chain)'].split(', ')

    # Hole die Sequenz von der PDB-Datenbank
    full_sequence = fetch_sequence_from_pdb(pdb_id)

    if full_sequence:
        # Extrahiere die Epitope-Positionen
        epitope_positions = [int(e.split('_')[0]) for e in epitope_list if e.split('_')[0].isdigit()]

        if epitope_positions:
            first_epitope = min(epitope_positions)
            start = max(0, first_epitope - 5)
            end = min(len(full_sequence), start + subseq_length)
            subsequence = full_sequence[start:end]

            # Auffüllen der Subsequenz auf die genaue Länge von 235
            if len(subsequence) < subseq_length:
                subsequence = subsequence + fetch_sequence_from_pdb(pdb_id)[:subseq_length - len(subsequence)]

            binary_epitope_string = map_epitopes_to_binary(subseq_length, epitope_positions, start)

        else:
            # Keine Epitope vorhanden, nehme die ersten subseq_length Stellen
            subsequence = full_sequence[:subseq_length]

            # Auffüllen der Subsequenz auf die genaue Länge von 235
            if len(subsequence) < subseq_length:
                subsequence = full_sequence + full_sequence[:subseq_length - len(full_sequence)]

            binary_epitope_string = '0' * subseq_length

        results.append({
            'PDB ID': pdb_id,
            'Subsequence': subsequence,
            'Binary Epitope': binary_epitope_string
        })
    else:
        # Fehlerfall behandeln, wenn die Sequenz nicht abgerufen werden konnte
        results.append({
            'PDB ID': pdb_id,
            'Subsequence': '',
            'Binary Epitope': ''
        })

# Ausgabe der Ergebnisse als DataFrame
results_df = pd.DataFrame(results)

# Speichere das DataFrame als CSV-Datei
results_df.to_csv('epitope_mapped_sequences.csv', index=False)

print("Die Datei wurde erfolgreich als 'epitope_mapped_sequences.csv' gespeichert.")