"""
This skript is for loading the data in the training pipeline step by step into memory instead of all at once.
Auth: Florian Zwicker
"""


# imports




class DataLoader:
    def __init__(self, structure_data_path, sequence_data_path):
        self.structure_data_path = structure_data_path
        self.sequence_data_path = sequence_data_path