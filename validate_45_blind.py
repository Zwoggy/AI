import pickle

from ai_functionality_old import load_model_and_tokenizer, modify_with_context, evaluate_model


def validate_on_45_blind():
    import pandas as pd
    import numpy as np

    # Assuming load_model_and_tokenizer() and evaluate_model() are defined elsewhere
    model, encoder = load_model_and_tokenizer()
    # Read the DataFrame with the sequences and the binary epitope
    df = pd.read_csv('./data/epitope3d_dataset_45_Blind_Test_manual_with_epitopes2.csv')
    # Prepare lists for modification
    epitope_list = [np.array([-1 if int(x) == 0 else int(x) for x in row['Epitope Sequence']]) for idx, row in df.iterrows()]

    print("epitope_list", epitope_list)
    antigen_list = [row['Sequence'] for idx, row in df.iterrows()]
    with open('./AI/tokenizer.pickle', 'rb') as handle:
        encoder = pickle.load(handle)
    antigen_list = encoder.texts_to_sequences(antigen_list)
    # Modify the sequences and get padded results
    modified_antigen_list, modified_epitope_list, padded_length = modify_with_context(epitope_list, antigen_list, length_of_longest_sequence=235) #ERROR HERE
    print("modified_epitope_list", modified_epitope_list[0])
    # Evaluate the model with modified sequences
    results = []
    for idx, (pdb_id, padded_sequence, true_binary_epitope) in enumerate(
            zip(df['PDB ID'], modified_antigen_list, modified_epitope_list)):
        # Convert the padded sequence back to a string if needed
        padded_sequence_str = ''.join([char for char in padded_sequence.astype(str)])

        # Calculate AUC, Recall, Precision, and F1
        recall, precision, f1 = evaluate_model(model, encoder, [padded_sequence_str], true_binary_epitope)

        # Store the results
        results.append({
            'PDB ID': pdb_id,
            # 'AUC': auc,
            'Recall': recall,
            'Precision': precision,
            'F1-Score': f1
        })
    # Save the results as a DataFrame
    results_df = pd.DataFrame(results)
    # Optionally: Save the results in a CSV
    results_df.to_csv('evaluation_results.csv', index=False)
    print("Evaluation completed and saved in 'evaluation_results.csv'.")
