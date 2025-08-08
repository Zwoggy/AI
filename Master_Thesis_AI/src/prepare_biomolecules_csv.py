import pandas as pd
import requests
import time

def fetch_sequence_from_pdb(pdb_id):
    """Fetch protein sequence from RCSB PDB in FASTA format."""
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Parse FASTA content and remove header line
            lines = response.text.strip().splitlines()
            sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
            return sequence if sequence else None
        else:
            print(f"Failed to fetch {pdb_id}: HTTP {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error fetching {pdb_id}: {e}")
        return None

def add_sequences_to_csv(input_csv, output_csv):
    # Load CSV file
    df = pd.read_csv(input_csv)

    if 'PDB ID' not in df.columns:
        raise ValueError("Input CSV must contain a 'PDB ID' column")

    # Fetch sequences
    print("Fetching sequences from RCSB PDB...")
    sequences = {}
    for pdb_id in df['PDB ID'].unique():
        seq = fetch_sequence_from_pdb(pdb_id)
        sequences[pdb_id] = seq
        time.sleep(0.1)  # to avoid hammering the server

    # Add sequences to DataFrame
    df['Sequence'] = df['PDB ID'].map(sequences)

    # Save updated CSV
    df.to_csv(output_csv, index=False)
    print(f"Done! Output written to: {output_csv}")

if __name__ == "__main__":
    import argparse
    try:
        parser = argparse.ArgumentParser(description="Add protein sequences from PDB to a CSV file.")
        parser.add_argument("input_csv", help="Path to input CSV file (must contain 'PDB ID' column)")
        parser.add_argument("output_csv", help="Path to save output CSV with added 'Sequence' column")
        args = parser.parse_args()

        add_sequences_to_csv(args.input_csv, args.output_csv)
    except:
        input_csv = "C:/Users/fkori/PycharmProjects/AI/data/Caroll_et_al_data/biomolecules.csv"
        output_csv = "C:/Users/fkori/PycharmProjects/AI/data/Caroll_et_al_data/biomolecules_incl_sequences.csv"
        add_sequences_to_csv(input_csv, output_csv)
