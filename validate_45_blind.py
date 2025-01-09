from ai_functionality_old import load_model_and_tokenizer, modify_with_context, evaluate_model


def validate_on_45_blind():
    import pandas as pd
    import numpy as np

    # Assuming load_model_and_tokenizer() and evaluate_model() are defined elsewhere
    model, encoder = load_model_and_tokenizer()
    # Read the DataFrame with the sequences and the binary epitope
    df = pd.read_csv('./AI/data/epitope3d_dataset_45_Blind_Test_translated.csv')  # Update with your actual file path
    # Prepare lists for modification
    epitope_list = [np.array([int(x) for x in row['Epitope 0/1 Sequence']]) for idx, row in df.iterrows()]
    antigen_list = [row['Raw Sequence'] for idx, row in df.iterrows()]
    # Modify the sequences and get padded results
    modified_antigen_list, modified_epitope_list, padded_length = modify_with_context(antigen_list, epitope_list, 235)
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
