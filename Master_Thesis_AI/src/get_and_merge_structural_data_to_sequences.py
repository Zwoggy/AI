import os
import json
import numpy as np
from Bio.PDB import MMCIFParser, DSSP

def build_structural_features(id_list, antigen_array):
    """
    Build structural features for each sequence ID and combine with antigen_list.

    Args:
        id_list (list of str): List of IDs, order must match antigen_list.
        antigen_array (np.ndarray): Tokenized, padded sequences, shape (N, max_len, seq_embed_dim).
        data_root (str): Path to folder containing folders per ID with AF files.

    Returns:
        X_structural: np.ndarray, shape (N, max_len, NUM_STRUCT_FEATURES)
        X_combined: np.ndarray, shape (N, max_len, seq_embed_dim + NUM_STRUCT_FEATURES)
    """
    # If your array is (N, max_len), expand it to (N, max_len, 1)
    if len(antigen_array.shape) == 2:
        antigen_array = np.expand_dims(antigen_array, axis=-1)

    data_root = "./data/BP3_Data/structures/folds/"
    NUM_STRUCT_FEATURES = 7  # SASA(1) + SS(3) + pLDDT(1) + PAE_mean(1) + Depth(1)
    N_samples, max_len, seq_embed_dim = antigen_array.shape

    assert len(id_list) == N_samples, "id_list length must match antigen_list sample count"

    X_structural = np.zeros((N_samples, max_len, NUM_STRUCT_FEATURES))

    parser = MMCIFParser()

    for idx, ID in enumerate(id_list):
        ID_on_disk = ID[:-1] + ID[-1].lower()
        print(f"Processing {ID} ({idx+1}/{N_samples})")

        cif_path = os.path.join(data_root, ID_on_disk, "/fold_", ID_on_disk, "_model_0.cif")
        json_path = os.path.join(data_root, ID_on_disk, "/fold_", ID_on_disk, "_full_data_0.json")

        if not (os.path.exists(cif_path) and os.path.exists(json_path)):
            print(f"⚠️ Missing files for {ID}, skipping...")
            continue

        # Load structure and DSSP
        structure = parser.get_structure(ID, cif_path)
        model = next(structure.get_models())
        dssp = DSSP(model, cif_path)

        # Extract SASA and SS per residue
        sasa, ss = [], []
        for key in dssp.keys():
            sasa.append(dssp[key][3])
            ss.append(dssp[key][2])

        # Load JSON pLDDT and PAE
        with open(json_path) as f:
            data = json.load(f)

        plddt = data["plddt"]
        pae = np.array(data["pae"])
        pae_mean = pae.mean(axis=1)

        # Compute residue depth as 1 - (relative SASA)
        max_sasa = max(sasa) if max(sasa) > 0 else 1.0
        depth = [1 - (s / max_sasa) for s in sasa]

        # One-hot encode SS: H, E, - (coil)
        SS_CODES = {"H": 0, "E": 1, "-": 2}
        ss_onehot = np.zeros((len(ss), 3))
        for i, s in enumerate(ss):
            idx_ss = SS_CODES.get(s, 2)
            ss_onehot[i, idx_ss] = 1

        # Build per-residue features padded/truncated to max_len
        X_struct = np.zeros((max_len, NUM_STRUCT_FEATURES))
        n_residues = min(len(sasa), max_len)

        for i in range(n_residues):
            X_struct[i, 0] = sasa[i]
            X_struct[i, 1:4] = ss_onehot[i]
            X_struct[i, 4] = plddt[i]
            X_struct[i, 5] = pae_mean[i]
            X_struct[i, 6] = depth[i]

        X_structural[idx] = X_struct

    # Combine sequence embeddings and structural features
    X_combined = np.concatenate([antigen_array, X_structural], axis=-1)
    print(f"Combined features shape: {X_combined.shape}")

    return X_structural, X_combined
