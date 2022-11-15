import json
import torch
from torch.utils.data import IterableDataset
import pandas as pd
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pickle


class DataGenerator(IterableDataset):
    """
    Class for generating batches of target protein affinity data.
    """

    def __init__(self, data_json_path, k):
        self.df = pd.DataFrame(json.load(open(data_json_path)))[:10]
        self.size = self.df.shape[0]
        self.k = k

        # Load pre-computed smiles embeddings
        with open('data/smiles_to_embeddings.pickle', 'rb') as f:
            self.smiles_embeddings = pickle.load(f)

    def _sample(self):
        """
        Returns a tuple containing:
            1. Batch of SMILES embeddings with shape (K+1, 2, 767)
            2. Batch of one-hot binary labels with shape (K+1, 2, 2)
        """
        # Find protein target with >= K positive and negative examples
        task = self.df.iloc[np.random.randint(self.size)]
        while (len(task['smiles_0']) < self.k+1) or (len(task['smiles_1']) < self.k+1):
            # Sample another random protein target
            task = self.df.iloc[np.random.randint(self.size)]

        # Sample K+1 positive and negative examples
        indices_0 = np.random.choice(len(task['smiles_0']), self.k+1, replace=False)
        indices_1 = np.random.choice(len(task['smiles_1']), self.k+1, replace=False)
        smiles_batch_0 = [task['smiles_0'][i][0] for i in indices_0]
        smiles_batch_1 = [task['smiles_1'][i][0] for i in indices_1]

        # Generate pretrained SMILES embeddings
        smiles_embeddings_0 = np.stack(list(map(lambda s: self.smiles_embeddings[s], smiles_batch_0)))
        smiles_embeddings_1 = np.stack(list(map(lambda s: self.smiles_embeddings[s], smiles_batch_1)))
        smiles_embeddings = np.stack([smiles_embeddings_0, smiles_embeddings_1], axis=1)

        # Assign labels
        labels = np.repeat(np.eye(2, 2)[None, :, :], self.k+1, axis=0)

        # Shuffle query set
        embeddings_and_labels = np.concatenate((smiles_embeddings, labels), axis=-1)
        np.random.shuffle(embeddings_and_labels[-1])
        smiles_embeddings = embeddings_and_labels[..., : -2]
        labels = embeddings_and_labels[..., -2 :]

        return (smiles_embeddings, labels)

    def __iter__(self):
        while True:
            yield self._sample()
