import pandas as pd
import re


# Funktion, um eine Sequenz von Nullen zu erzeugen und an Epitop-Positionen Einsen zu setzen
def create_epitope_sequence(epitope_list, sequence_length):
    sequence = [0] * sequence_length
    positions = [int(re.findall(r'\d+', ep)[0]) for ep in epitope_list]

    for pos in positions:
        if pos < sequence_length:
            sequence[pos] = 1

    return ''.join(map(str, sequence))


# Einlesen der CSV-Datei
df = pd.read_csv('epitope3d_dataset_45_Blind_Test.csv')

# Bestimmen der maximalen Sequenzlänge
max_length = 0
for epitopes in df['Epitope List (residueid_residuename_chain)']:
    positions = [int(re.findall(r'\d+', ep)[0]) for ep in epitopes.split(',')]
    max_length = max(max_length, max(positions) + 1)

# Erstellen der Epitopsequenzen und Ersetzen der ID durch die Sequenz
df['PDB ID'] = df['Epitope List (residueid_residuename_chain)'].apply(
    lambda x: create_epitope_sequence(x.split(','), max_length)
)

# Entfernen der alten Epitopen-Liste
df = df.drop(columns=['Epitope List (residueid_residuename_chain)'])

# Speichern des Ergebnisses in einer neuen Datei
df.to_csv('epitope_sequences_with_ids.csv', index=False)

print("Die Epitopsequenzen wurden erfolgreich in 'epitope_sequences_with_ids.csv' gespeichert.")