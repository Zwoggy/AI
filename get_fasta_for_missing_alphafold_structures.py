import os
import requests

# === Parameter ===
missing_file = "C:/Users/fkori/PycharmProjects/AI/data/missing_structures.txt"  # Datei mit fehlenden UniProt-IDs
output_folder = "C:/Users/fkori/PycharmProjects/AI/data/fasta_sequences"
os.makedirs(output_folder, exist_ok=True)


# === Funktionen ===
def download_fasta(uniprot_id):
    """
    Lädt die FASTA-Sequenz für eine UniProt-ID herunter und speichert sie.
    """
    output_path = os.path.join(output_folder, f"{uniprot_id}.fasta")
    if os.path.exists(output_path):
        print(f"FASTA file already exists: {output_path}")
        return True  # Datei existiert bereits

    base_url = "https://www.uniprot.org/uniprotkb"
    fasta_url = f"{base_url}/{uniprot_id}.fasta"

    try:
        response = requests.get(fasta_url)
        if response.status_code == 200:
            with open(output_path, "w") as file:
                file.write(response.text)
            print(f"Saved FASTA file: {output_path}")
            return True
        else:
            print(f"FASTA file not found for UniProt ID {uniprot_id}")
            return False
    except Exception as e:
        print(f"Error downloading FASTA for {uniprot_id}: {e}")
        return False


# === Hauptprogramm ===
if __name__ == "__main__":
    # 1. Fehlende IDs laden
    if not os.path.exists(missing_file):
        print(f"Missing structures file not found: {missing_file}")
        exit(1)

    print("Loading missing UniProt IDs...")
    with open(missing_file, "r") as file:
        missing_ids = [line.strip() for line in file if line.strip()]

    # 2. FASTA-Dateien herunterladen
    for uniprot_id in missing_ids:
        print(f"Processing UniProt ID (FASTA): {uniprot_id}")
        download_fasta(uniprot_id)

    print("\n=== Download Complete ===")
    print(f"FASTA files saved to '{output_folder}'.")
