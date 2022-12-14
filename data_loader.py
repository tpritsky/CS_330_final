import json
import gzip
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM


class DataGenerator(IterableDataset):

    def __init__(self, data_json_path, k, repr, q=1):
        self.df = pd.DataFrame(json.load(open(data_json_path)))
        self.size = self.df.shape[0]
        self.k = k  # number of support examples per class in a task
        self.q = q  # number of query examples per class in a task - added for protonet
        self.repr = repr

        # Load pre-computed smiles embeddings
        with open('data/smiles_to_embeddings.pickle', 'rb') as f:
            self.smiles_embeddings = pickle.load(f)
        
        # Load and preprocess pre-computed ESM protein embeddings
        with gzip.open('embeddings/esm2_t30_150M_UR50D.json.gz', 'r') as f:
            esm2 = json.loads(f.read().decode('utf-8'))
        self.protein_embeddings = {label[8:]:np.array(sample) for label, sample in zip(esm2['labels'],  esm2['samples'])}
            # keys:     task number (string) ex. '54'
            # values:   protein embedding for protein task (np.array of shape (640,))
        
        # Load and preprocess pre-computed VAE protein embeddings
        with open('embeddings/vae_100_esm2_t30_150M_UR50D_embeddings.p', 'rb') as f:
            vae_prot = pickle.load(f)
        self.vae_protein_embeddings = {label:sample for label, sample in zip(vae_prot['labels'],  vae_prot['samples'])}
            # keys:     task number (string) ex. '54'
            # values:   protein embedding for protein task (np.array of shape (100,))

    def _sample(self):
        # Find protein target with >= K positive and negative examples
        task = self.df.iloc[np.random.randint(self.size)]
        while (len(task['smiles_0']) < self.k+self.q) or (len(task['smiles_1']) < self.k+self.q):
            # Sample another random protein target
            task = self.df.iloc[np.random.randint(self.size)]

        # Sample K+Q positive and negative examples
        indices_0 = np.random.choice(len(task['smiles_0']), self.k+self.q, replace=False)
        indices_1 = np.random.choice(len(task['smiles_1']), self.k+self.q, replace=False)
        smiles_batch_0 = [task['smiles_0'][i][0] for i in indices_0]
        smiles_batch_1 = [task['smiles_1'][i][0] for i in indices_1]

        # Generate pretrained SMILES embeddings
        smiles_embeddings_0 = np.stack(list(map(lambda s: self.smiles_embeddings[s], smiles_batch_0)))
        smiles_embeddings_1 = np.stack(list(map(lambda s: self.smiles_embeddings[s], smiles_batch_1)))
        smiles_embeddings = np.stack([smiles_embeddings_0, smiles_embeddings_1], axis=1)  # (K+Q, N, embedding_dim)

        # Assign labels
        labels = np.repeat(np.eye(2, 2)[None, :, :], self.k+self.q, axis=0)  # (K+Q, N, N)

        # Shuffle query set
        if self.repr in ("concat", "concat_after_full"):
            protein_embedding = np.tile(self.protein_embeddings[task.name], (self.k+self.q, 2, 1))
                # tile to get shape (self.k+self.q, 2, prot_embed_dim) for concatentation
            embeddings_and_labels = np.concatenate((smiles_embeddings, protein_embedding, labels), axis=-1)
        elif self.repr in ("concat_smiles_vaeprot", "concat_after"):
            protein_embedding = np.tile(self.vae_protein_embeddings[int(task.name)], (self.k+self.q, 2, 1))
                # tile to get shape (self.k+self.q, 2, prot_embed_dim) for concatentation
            embeddings_and_labels = np.concatenate((smiles_embeddings, protein_embedding, labels), axis=-1)            
        else:  # "smiles_only"
            embeddings_and_labels = np.concatenate((smiles_embeddings, labels), axis=-1)  # (K+Q, N, embedding_dim+N)
        
        query_embeddings_labels = embeddings_and_labels[-self.q:] # (Q, N, embed_dim), view of embeddings_and_labels[-self.q:]
        for q in query_embeddings_labels:
            np.random.shuffle(q) # shuffles axis 0 corresponding to N
            # shuffle along axis for N independently for each query shot (contains 1 example for each of N classes)
            # do this shuffle order in which classes are fed in to prevent memorization

        embeddings = embeddings_and_labels[..., :-2]  # N=2, embeddings ~ (K+Q, N, embedding_dim)
        labels = embeddings_and_labels[..., -2:]  # N=2, labels ~ (K+Q, N, N)

        return embeddings, labels

    def __iter__(self):
        while True:
            yield self._sample()
