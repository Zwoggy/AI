"""
This script is currently to be run on a local machine.
Providing an input file with the output of the evaluation steps to plot the trends.

Auth: Florian Zwicker
"""


import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_blocks(filepath):
    """Parse your input file into a list of (model, hidden_units, DataFrame)"""
    blocks = []
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by double newlines
    raw_blocks = re.split(r'\n\s*\n', content.strip())

    current_model = None
    current_units = None
    current_csv = []

    for block in raw_blocks:
        lines = block.strip().splitlines()
        if len(lines) == 0:
            continue
        header = lines[0]
        match = re.match(r'(\d+)\s+([^\s]+)\s+with\s+(\d+)\s+hidden units', header)
        if match:
            if current_model:
                # Save previous block
                df = pd.read_csv(pd.compat.StringIO('\n'.join(current_csv)))
                blocks.append((current_model, current_units, df))
                current_csv = []

            current_model = match.group(2)
            current_units = int(match.group(3))
        else:
            current_csv.extend(lines)

    if current_model and current_csv:
        df = pd.read_csv(pd.compat.StringIO('\n'.join(current_csv)))
        blocks.append((current_model, current_units, df))

    return blocks


def generate_html_table(df, train_name='train', test_name='test'):
    """Return HTML table with colors & bold max per split"""
    html = []
    html.append('<table>')
    html.append('<thead><tr style="color:#e0e0e0;">')
    html.append('<th>Fold</th><th>Split</th><th>AUC</th><th>F1</th><th>Precision</th><th>Recall</th></tr></thead>')
    html.append('<tbody>')

    # Split
    for split in df['split'].unique():
        df_split = df[df['split'] == split]
        # Find max per metric
        max_vals = {
            'auc': df_split['auc'].max(),
            'f1': df_split['f1'].max(),
            'precision': df_split['precision'].max(),
            'recall': df_split['recall'].max(),
        }

        for _, row in df_split.iterrows():
            bg = '#004d4d' if split == train_name else '#3d5c3d'
            html.append(f'<tr style="background-color:{bg}; color:#e0e0e0;">')
            html.append(f'<td>{int(row["fold"])}</td>')
            html.append(f'<td>{row["split"]}</td>')

            for metric in ['auc', 'f1', 'precision', 'recall']:
                val = float(row[metric])
                val_fmt = f'{val:.4f}'
                if abs(val - max_vals[metric]) < 1e-8:
                    html.append(f'<td><b>{val_fmt}</b></td>')
                else:
                    html.append(f'<td>{val_fmt}</td>')
            html.append('</tr>')

    html.append('</tbody></table>')
    return '\n'.join(html)


def plot_trends(blocks, metric='auc', train_name='train', test_name='test'):
    """Plot train vs test metric trends vs hidden units"""
    hidden_units = []
    train_means = []
    test_means = []

    for _, units, df in blocks:
        hidden_units.append(units)
        train_val = df[df['split'] == train_name][metric].mean()
        test_val = df[df['split'] == test_name][metric].mean()
        train_means.append(train_val)
        test_means.append(test_val)

    plt.figure(figsize=(10, 6))
    plt.plot(hidden_units, train_means, marker='o', label='Train')
    plt.plot(hidden_units, test_means, marker='o', label='Test')
    plt.title(f'{metric.upper()} Trend vs Hidden Units')
    plt.xlabel('Hidden Units')
    plt.ylabel(metric.upper())
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    input_file = 'your_file.txt'
    blocks = parse_blocks(input_file)

    for model, units, df in blocks:
        print(f'\nModel: {model} with {units} hidden units\n')
        html = generate_html_table(df)
        print(html)

    plot_trends(blocks, metric='auc')
