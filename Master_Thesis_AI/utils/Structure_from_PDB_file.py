"""
This skripts purpose is to retrieve and save the struktural data from the PDB file and store it as dataset.
Auth: Florian Zwicker
"""
import os
from Bio.PDB import PDBParser

import os
import numpy as np
from Bio.PDB import PDBParser, is_aa
from Bio.SeqUtils import seq1
import pickle

def extract_structure_data(pdb_dir, output_file):
    parser = PDBParser(QUIET=True)
    data = []

    for file in os.listdir(pdb_dir):
        if not file.endswith('_alphafold.pdb'):
            continue
        pdb_id = file.split('_')[0]
        filepath = os.path.join(pdb_dir, file)
        structure = parser.get_structure(pdb_id, filepath)

        coords = []
        sequence = ""
        ca_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if "CA" in residue:
                        ca = residue["CA"]
                        ca_coords.append(ca.get_coord())

                        sequence += seq1(residue.get_resname())
        coords = np.array(coords)
        if len(coords) < 2:
            continue
        print(f"Verarbeite Datei: {pdb_id}")
        print(f"Extrahierte Koordinaten: {len(ca_coords)}")
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

        data.append({
            'id': pdb_id,
            'sequence': sequence,
            'structure_array': dist_matrix  # (L, L) - Conv2D-ready
        })

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)



if __name__=='__main__':
    # Beispielaufruf:

    extract_structure_data(
        pdb_dir="/home/fzwicker/Forschungsprojekt_02/fasta_data/alphafold_output/",
        output_file="/home/fzwicker/Forschungsprojekt_02/git_project/data/alphafold_structures_conv2d.pkl"
    )

    with open("/home/fzwicker/Forschungsprojekt_02/git_project/data/alphafold_structures_conv2d.pkl", "rb") as f:
        data = pickle.load(f)

    print("Anzahl Einträge:", len(data))
    print("Beispiel-Eintrag:")
    print(data[0].keys())

