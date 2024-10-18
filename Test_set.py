import pandas as pd





#aminoacid_code = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M",
                  #"PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "SEM": "X"}




import pandas as pd

aminoacid_code = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "SEM": "X"
}

# Die Daten in der CSV-Datei einlesen
data = pd.read_csv("data/epitope3d_dataset_45_Blind_Test_manual.csv")


def convert_epitope_list(epitope_list):
    """
    Wandelt die Epitop-Liste in eine Liste von Tupeln (position, aminosäure)
    """
    converted_list = []
    for epitope in epitope_list.split(", "):
        pos, aa, _ = epitope.split("_")
        if aa in aminoacid_code:
            converted_list.append((int(pos), aminoacid_code[aa]))
    return converted_list


def create_epitope_sequence(epitope_list, sequence):
    """
    Erstellt eine Epitop-Sequenz aus 0en und markiert die Positionen der Epitope mit 1en
    """
    epitope_sequence = [0] * len(sequence)
    start_indices = [i for i in range(len(sequence)) if sequence[i] == epitope_list[0][1]]

    for start_index in start_indices:
        current_index = start_index
        is_correct = True
        positions = [start_index]

        for i in range(1, len(epitope_list)):
            expected_index = current_index + (epitope_list[i][0] - epitope_list[i - 1][0])
            if expected_index >= len(sequence) or sequence[expected_index] != epitope_list[i][1]:
                is_correct = False
                break
            positions.append(expected_index)
            current_index = expected_index

        if is_correct:
            for pos in positions:
                epitope_sequence[pos] = 1
            break

    return epitope_sequence


# Neue Spalte für die Epitop-Sequenzen hinzufügen
data["Epitope Sequence"] = ""

for index, row in data.iterrows():
    pdb_id = row["PDB ID"]
    epitope_list = row["Epitope List (residueid_residuename_chain)"]
    sequence = row["Sequence"].replace("(", "").replace(")","")

    converted_epitope_list = convert_epitope_list(epitope_list)
    epitope_seq = create_epitope_sequence(converted_epitope_list, sequence)

    # Epitop-Sequenz in die neue Spalte einfügen
    data.at[index, "Epitope Sequence"] = ''.join(map(str, epitope_seq))
    data.at[index, "Sequence"] = sequence

# Neue Datei speichern
data.to_csv("epitope3d_dataset_45_Blind_Test_manual_with_epitopes2.csv", index=False)