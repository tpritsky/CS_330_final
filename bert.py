import argparse
import random
import json

import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from data_loader import DataGenerator
from tqdm import tqdm

from transformers import BertConfig
from bert_utils import SmilesBertModel

def main(config):
    print(config)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # wandb.init(project="fs_bindingdb", entity="davidekuo")
        # wandb.config.update(config)
    else:
        device = torch.device("cpu")

    writer = SummaryWriter(
        f"runs/bert_{config.repr}_{config.dataset}_N{config.num_classes}_K{config.num_shot}"
        f"_Seed{config.random_seed}_HiddenDim{config.hidden_dim}_LR{config.learning_rate}_Dropout{config.dropout}"
    )

    # Create Data Generator
    train_iterable = DataGenerator(
        data_json_path=f'data/train.json',
        k=config.num_shot,
        repr=config.repr,
    )
    train_loader = iter(
        torch.utils.data.DataLoader(
            train_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    test_iterable = DataGenerator(
        data_json_path=f'data/val.json',
        k=config.num_shot,
        repr=config.repr,
    )
    test_loader = iter(
        torch.utils.data.DataLoader(
            test_iterable,
            batch_size=config.meta_batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
        )
    )

    repr_to_input_dims = {"smiles_only": config.num_classes + 767, 
                          "concat": config.num_classes + 767 + 640,
                          "concat_smiles_vaeprot": config.num_classes + 767 + 100}
    # smiles_embedding_dim = 767, protein_embedding_dim = 640, vae_protein_embedding_dim = 100

    # Create model
    model_config = BertConfig(
        max_position_embeddings = (config.num_shot+1)*config.num_classes,
        hidden_size = repr_to_input_dims[config.repr],
        num_hidden_layers = 4,
        num_attention_heads = 1,
        intermediate_size = 64,
        classifier_dropout = 0.3,
        attention_probs_dropout_prob = 0.3,
        hidden_dropout_prob = 0.3,
        k = config.num_shot,
        batch_size = config.meta_batch_size,
    )

    # Create model
    model = SmilesBertModel(model_config)
    model.to(device)

    # Create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    import time

    times = []
    best_val_acc = 0
    for step in tqdm(range(config.train_steps)):
        ## Sample Batch
        t0 = time.time()
        i, l = next(train_loader)
        i, l = i.to(device), l.to(device)
        t1 = time.time()

        ## Train
        # _, ls = train_step(i, l, model, optim)
        attention_mask = torch.ones((config.meta_batch_size, (config.num_shot+1)*config.num_classes))
        attention_mask = attention_mask.to(device)
        ls, _ = model(i.float(), l.float(), attention_mask)
        ls.backward()
        optim.step()
        optim.zero_grad()

        t2 = time.time()
        writer.add_scalar("Loss/train", ls, step)
        times.append([t1 - t0, t2 - t1])
        # if device == torch.device("cuda"):
        #     wandb.log({"Loss/train": ls})

        ## Evaluate
        if (step + 1) % config.eval_freq == 0:
            print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)
            i, l = next(test_loader)
            i, l = i.to(device), l.to(device)
            model.eval()
            tls, pred = model(i.float(), l.float(), attention_mask)

            print(
                "Train Loss:",
                ls.detach().cpu().numpy(),
                "Test Loss:",
                tls.detach().cpu().numpy(),
            )
            writer.add_scalar("Loss/test", tls, step)

            # pred = F.sigmoid(pred) # (batch, n, n)
            
            pred = torch.argmax(pred, axis=1)  # could be error prone

            l = torch.argmax(l[:, -1, :, :], axis=1)

            acc = pred.eq(l).sum().item() / (
                config.meta_batch_size * config.num_classes
            )
            print("Val Accuracy", acc)
            writer.add_scalar("Accuracy/val", acc, step)

            if acc > best_val_acc:
                torch.save(model, 'model/bert_model.pt')
                print("Saved model.")
                best_val_acc = acc


            times = np.array(times)
            print(
                f"Sample time {times[:, 0].mean()} Train time {times[:, 1].mean()}"
            )
            # if device == torch.device("cuda"):
            #     wandb.log({"Loss/test": tls})
            #     wandb.log({"Accuracy/test": acc})
            #     wandb.log({"Sample time": times[:, 0].mean(), "Train time": times[:, 1].mean()})

            times = []

            model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_shot", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=500)
    parser.add_argument("--meta_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_steps", type=int, default=25000)
    parser.add_argument("--repr", type=str, default="smiles_only")  # alternatively "smiles_only", "concat", "vaesmiles_only"
    parser.add_argument("--dataset", type=str, default="full")  # alternatively "full"
    parser.add_argument("--dropout", type=float, default=0.35)
    main(parser.parse_args())
