"""
This skript is for loading the data in the training pipeline step by step into memory instead of all at once.
Auth: Florian Zwicker
"""


# imports


from tf_keras.utils import Sequence
import numpy as np


class EpitopeDataGenerator(Sequence):
    def __init__(self, inputs, labels, sample_weights=None, batch_size=32, shuffle=True):
        self.inputs = inputs
        self.labels = labels
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.inputs))
        self.on_epoch_end()

    def __len__(self):
        # Anzahl Batches pro Epoche
        return int(np.ceil(len(self.inputs) / self.batch_size))

    def __getitem__(self, index):
        # Gibt einen Batch zurück
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        batch_inputs = self.inputs[batch_indices]
        batch_labels = self.labels[batch_indices]

        if self.sample_weights is not None:
            batch_weights = self.sample_weights[batch_indices]
            return batch_inputs, batch_labels, batch_weights
        else:
            return batch_inputs, batch_labels

    def on_epoch_end(self):
        # Mische die Daten am Ende jeder Epoche
        if self.shuffle:
            np.random.shuffle(self.indices)
