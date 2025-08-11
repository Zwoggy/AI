"""
A utility module for processing epitope data and adding binary epitope
information to a CSV file.

This module provides functionality for reading sequence data from an input CSV
file, computing binary epitope representations, and saving the updated data
to an output CSV file.

Author: Florian Zwicker
"""





import pandas as pd
import json





def add_epitopes(input_csv, output_csv):
    # Load CSV file
    df = pd.read_csv(input_csv)
    df.insert(0, "Binary Epitop", None)
    for index, row in df.iterrows():
        sequence_length = len(row["Sequence"])
        binary_epitope_list = [int(0)] * sequence_length
        current_positive_epitopes = row["Epitope"].split("+")
        start_index = int(row["Starting_Index"])
        for epitope in current_positive_epitopes:
            epitope = int(epitope.strip())
            pos = epitope - start_index
            print("ID and pos: ", row["PDB ID"], pos)
            binary_epitope_list[pos] = 1

        df.loc[index, "Binary Epitop"] = json.dumps(binary_epitope_list)

    df.to_csv(output_csv, index=False)



if __name__ == "__main__":
    input_csv = "C:/Users/fkori/PycharmProjects/AI/data/Caroll_et_al_data/biomolecules_incl_sequences.csv"
    output_csv = "C:/Users/fkori/PycharmProjects/AI/data/Caroll_et_al_data/biomolecules_incl_sequences_and_epitopes.csv"
    add_epitopes(input_csv, output_csv)
