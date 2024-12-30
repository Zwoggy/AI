import os
import requests
import pandas as pd

# === Parameter ===
# Pfad zur Excel-Datei mit den Sequenzen
input_file = "C:/Users/fkori/PycharmProjects/AI/data/Dataset.xlsx"
uniprot_column = "Accession"   # Spalte mit den UniProt-IDs


# Ordner für AlphaFold-Strukturen
output_folder = "C:/Users/fkori/PycharmProjects/AI/data/alphafold_structures_02"
os.makedirs(output_folder, exist_ok=True)

missing_alphafold = []
missing_all = []


# === Funktionen ===
def download_alphafold_structure(uniprot_id):
    """
    Lädt eine AlphaFold-Struktur herunter, falls vorhanden.
    """
    output_path = os.path.join(output_folder, f"{uniprot_id}_alphafold.pdb")
    if os.path.exists(output_path):
        print(f"AlphaFold structure already exists: {output_path}")
        return True  # Datei existiert bereits

    base_url = "https://alphafold.ebi.ac.uk/files"
    pdb_url = f"{base_url}/AF-{uniprot_id}-F1-model_v4.pdb"

    try:
        response = requests.get(pdb_url)
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                file.write(response.content)
            print(f"Saved AlphaFold structure: {output_path}")
            return True
        else:
            print(f"AlphaFold structure not found for UniProt ID {uniprot_id}")
            return False
    except Exception as e:
        print(f"Error downloading AlphaFold structure for {uniprot_id}: {e}")
        return False


def get_pdb_ids_from_uniprot(uniprot_id):
    """
    Holt PDB-IDs, die mit einer UniProt-ID verknüpft sind.
    """
    url = f"https://www.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        pdb_ids = []
        for xref in data.get("dbReferences", []):
            if xref["type"] == "PDB":
                pdb_ids.append(xref["id"])
        return pdb_ids
    except Exception as e:
        print(f"Error fetching PDB IDs for UniProt ID {uniprot_id}: {e}")
        return []


def download_pdb_structure(pdb_id):
    """
    Lädt eine PDB-Struktur herunter und speichert sie lokal.
    """
    output_path = os.path.join(output_folder, f"{pdb_id}.pdb")
    if os.path.exists(output_path):
        print(f"PDB structure already exists: {output_path}")
        return True  # Datei existiert bereits

    base_url = "https://files.rcsb.org/download"
    pdb_url = f"{base_url}/{pdb_id}.pdb"

    try:
        response = requests.get(pdb_url)
        if response.status_code == 200:
            with open(output_path, "wb") as file:
                file.write(response.content)
            print(f"Saved PDB structure: {output_path}")
            return True
        else:
            print(f"PDB structure not found for ID {pdb_id}")
            return False
    except Exception as e:
        print(f"Error downloading PDB structure for {pdb_id}: {e}")
        return False


# === Hauptprogramm ===
if __name__ == "__main__":
    # 1. Excel-Datei laden
    print("Loading UniProt IDs from Excel...")
    df = pd.read_excel(input_file)
    uniprot_ids = df[uniprot_column].dropna().unique()

    # 2. AlphaFold-Strukturen herunterladen
    for uniprot_id in uniprot_ids:
        print(f"Processing UniProt ID (AlphaFold): {uniprot_id}")
        if not download_alphafold_structure(uniprot_id):
            missing_alphafold.append(uniprot_id)

    # 3. PDB-Strukturen für fehlende AlphaFold-IDs herunterladen
    for uniprot_id in missing_alphafold:
        print(f"Processing UniProt ID (PDB): {uniprot_id}")
        pdb_ids = get_pdb_ids_from_uniprot(uniprot_id)
        if pdb_ids:
            for pdb_id in pdb_ids:
                if download_pdb_structure(pdb_id):
                    break  # Falls eine PDB-Struktur gefunden wurde
        else:
            missing_all.append(uniprot_id)

    # 4. Ergebnis
    print("\n=== Download Summary ===")
    print(f"AlphaFold structures missing: {len(missing_alphafold)}")
    print(f"Completely missing structures: {len(missing_all)}")
    print("IDs with no available structure:")
    print(missing_all)

    # 5. Fehlende IDs in Datei speichern
    with open("C:/Users/fkori/PycharmProjects/AI/data/missing_structures.txt", "w") as log_file:
        log_file.write("\n".join(missing_all))
    print(f"Missing structure IDs saved to 'C:/Users/fkori/PycharmProjects/AI/data/missing_structures.txt'.")