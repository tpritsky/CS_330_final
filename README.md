# CS330 Final Project
## Meta-learning for binding affinity prediction

### Code
- `bert.py`: code for BERT-based black-box meta-learner
- `lstm.py`: code for LSTM-based black-box meta-learner
- `protonet.py`: code for prototypical network meta-learner
- `data_loader.py`: code for dataloader used for both BERT and LSTM models
- `evaluation.ipynb`: IPython Notebook for evaluating a trained BERT or LSTM model
- `precompute_smiles_embeddings.ipynb`: IPython Notebook for precomputing and storing SMILES embeddings

### Data / Files
- `/data/`: training, validation, and test datasets stored as JSON files
- `/embeddings/`: precomputed protein embeddings stored as JSON and pickle files
