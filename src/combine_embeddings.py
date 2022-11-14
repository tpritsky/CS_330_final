import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import os 
import json
import gzip
import sys

class MyDataset(Dataset):
   def __init__(self, root):
        self.root = root
        self.files = os.listdir(root) # take all files in the root directory
   def __len__(self):
        return len(self.files)
   def __getitem__(self, idx):
        result_dict = torch.load(os.path.join(self.root, self.files[idx])) # load the features of this sample
        label = result_dict['label']
        sample = result_dict['mean_representations'][30]
        return sample, label

#load data
dataset = MyDataset(sys.argv[1])

#create dictionary
samples = []
labels = []
for i in range(len(dataset)):
    sample, label = dataset[i]
    samples.append(sample.tolist())
    labels.append(label)

#generate embeddings
embeddings = {'labels':labels, 'samples': samples}

#export to gzip
with gzip.open('data/sample_output.json.gz','wt', encoding='utf-8') as outfile:
    json.dump(embeddings, outfile)


