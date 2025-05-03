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
import  Bio.PDB as PDB

def is_aa(residue):
    # Diese Funktion überprüft, ob es sich um eine Aminosäure handelt.
    return residue.id[0] == " "  # Nur natürliche Aminosäuren

def extract_structure_data(input_dir, output_file):
    parser = PDB.PDBParser(QUIET=True)
    all_data = []
    num_files = 0
    num_valid = 0

    # 1. Direkt im input_dir liegende .pdb-Dateien verarbeiten
    for file in os.listdir(input_dir):
        if file.endswith(".pdb"):
            pdb_path = os.path.join(input_dir, file)
            pdb_id = os.path.splitext(file)[0]
            try:
                structure = parser.get_structure(pdb_id, pdb_path)
                model = structure[0]
                ca_coords = []
                sequence = ""
                for chain in model:
                    for residue in chain:
                        if "CA" in residue:
                            ca = residue["CA"]
                            ca_coords.append(ca.get_coord())
                        if is_aa(residue):  # Nur Aminosäuren (keine Wasserstoffe oder Liganden)
                            sequence += seq1(residue.get_resname())  # Aminosäuresequenz hinzufügen
                if len(ca_coords) == 0:
                    print(f"⚠️  Keine CA-Koordinaten in: {pdb_path}")
                    continue
                ca_array = np.array(ca_coords)
                all_data.append({
                    "id": pdb_id,
                    "structure_array": ca_array,
                    "sequence": sequence,  # Sequenz hinzufügen
                })
                num_valid += 1
            except Exception as e:
                print(f"❌ Fehler bei Datei {pdb_path}: {e}")
            num_files += 1

    # 2. ranked_0.pdb-Dateien aus Unterverzeichnissen wie gehabt
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file == "ranked_0.pdb":
                pdb_path = os.path.join(root, file)
                pdb_id = os.path.basename(root)
                try:
                    structure = parser.get_structure(pdb_id, pdb_path)
                    model = structure[0]
                    ca_coords = []
                    sequence = ""
                    for chain in model:
                        for residue in chain:
                            if "CA" in residue:
                                ca = residue["CA"]
                                ca_coords.append(ca.get_coord())
                            if is_aa(residue):  # Nur Aminosäuren
                                sequence += seq1(residue.get_resname())  # Aminosäuresequenz hinzufügen
                    if len(ca_coords) == 0:
                        print(f"⚠️  Keine CA-Koordinaten in: {pdb_path}")
                        continue
                    ca_array = np.array(ca_coords)
                    all_data.append({
                        "id": pdb_id,
                        "structure_array": ca_array,
                        "sequence": sequence,  # Sequenz hinzufügen
                    })
                    num_valid += 1
                except Exception as e:
                    print(f"❌ Fehler bei Datei {pdb_path}: {e}")
                num_files += 1

            # Falls ranked_0.pdb fehlt, versuche das erste unrelaxed_model_X_pred_0.pdb zu verwenden
            elif file.startswith("unrelaxed_model") and file.endswith("pred_0.pdb") and "ranked_0.pdb" not in files:
                pdb_path = os.path.join(root, file)
                pdb_id = os.path.basename(root)
                try:
                    structure = parser.get_structure(pdb_id, pdb_path)
                    model = structure[0]
                    ca_coords = []
                    sequence = ""
                    for chain in model:
                        for residue in chain:
                            if "CA" in residue:
                                ca = residue["CA"]
                                ca_coords.append(ca.get_coord())
                            if is_aa(residue):  # Nur Aminosäuren
                                sequence += seq1(residue.get_resname())  # Aminosäuresequenz hinzufügen
                    if len(ca_coords) == 0:
                        print(f"⚠️  Keine CA-Koordinaten in: {pdb_path}")
                        continue
                    ca_array = np.array(ca_coords)
                    all_data.append({
                        "id": pdb_id,
                        "structure_array": ca_array,
                        "sequence": sequence,  # Sequenz hinzufügen
                    })
                    num_valid += 1
                except Exception as e:
                    print(f"❌ Fehler bei Datei {pdb_path}: {e}")
                num_files += 1

    print(f"\n📊 Verarbeitete Dateien: {num_files}")
    print(f"✅ Erfolgreich extrahiert: {num_valid}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(all_data, f)




if __name__=='__main__':
    # Beispielaufruf:

    extract_structure_data(
        input_dir="/home/fzwicker/Forschungsprojekt_02/fasta_data/alphafold_output/",
        output_file="/home/fzwicker/Forschungsprojekt_02/git_project/data/alphafold_structures_conv2d.pkl"
    )

    with open("/home/fzwicker/Forschungsprojekt_02/git_project/data/alphafold_structures_conv2d.pkl", "rb") as f:
        data = pickle.load(f)

    print("Anzahl Einträge:", len(data))
    print("Beispiel-Eintrag:")
    print(data[0].keys())

