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
        # Indizes für den aktuellen Batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        # Daten herausholen
        batch_inputs = self.inputs[batch_indices]
        batch_labels = self.labels[batch_indices]

        # Falls Input-Shape (B, 1, L, D): Squeeze zweite Dimension
        if batch_inputs.ndim == 4 and batch_inputs.shape[1] == 1:
            batch_inputs = np.squeeze(batch_inputs, axis=1)  # → (B, L, D)

        # Optional: Sample Weights
        """
        if self.sample_weights is not None:
            batch_weights = self.sample_weights[batch_indices]
            return batch_inputs, batch_labels, batch_weights
        """
        return batch_inputs, batch_labels

    def on_epoch_end(self):
        # Mische die Daten am Ende jeder Epoche
        if self.shuffle:
            np.random.shuffle(self.indices)

