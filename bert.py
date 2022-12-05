import argparse
import random
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from data_loader import DataGenerator
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple
from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertPreTrainedModel,
)


class SmilesBertModel(BertPreTrainedModel):
    """Multi-headed BERT-based model for SMILES sequences."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.k = config.k
        self.batch_size = config.batch_size
        self.repr = config.repr

        self.encoder = BertEncoder(config)
        if self.repr == "concat_after":
            self.linear = nn.Linear(config.hidden_size + 100, 2)
        elif self.repr == "concat_after_full":
            self.linear = nn.Linear(config.hidden_size + 640, 2)
        else:
            self.linear = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_images: torch.Tensor,
        raw_input_labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, (input_images.shape[0], input_images.shape[-1]))

        input_labels = raw_input_labels.clone()
        input_labels[:, -1, :, :] = 0

        if self.repr == "concat_after":
            protein_embeds = input_images[:, :, :, -100:].float()
            input_images = input_images[:, :, :, :-100]
        if self.repr == "concat_after_full":
            protein_embeds = input_images[:, :, :, -640:].float()
            input_images = input_images[:, :, :, :-640]

        input_images_and_labels = torch.cat((input_images, input_labels), -1)
        B, K_1, N, D = input_images_and_labels.shape
        input_images_and_labels = input_images_and_labels.reshape((B, -1, D))

        encoder_outputs = self.encoder(
            input_images_and_labels,  # (batch, seq, dim)
            attention_mask=extended_attention_mask,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=self.config.use_cache,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]

        if self.repr == "concat_after":
            protein_embeds = protein_embeds.reshape((B, -1, 100))
            sequence_output = torch.concat((sequence_output, protein_embeds), axis=-1)  
        if self.repr == "concat_after_full":
            protein_embeds = protein_embeds.reshape((B, -1, 640))
            sequence_output = torch.concat((sequence_output, protein_embeds), axis=-1)

        sequence_output = torch.reshape(sequence_output,[self.batch_size, self.k + 1, 2, -1])
        query_embeddings = sequence_output[:, -1, :, :]
        query_labels = raw_input_labels[:, -1, :, :]

        pred = self.linear(query_embeddings)
        pred = F.sigmoid(pred)

        loss = F.binary_cross_entropy(pred, query_labels)

        return loss, pred


def main(config):
    print(config)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    writer = SummaryWriter(
        f"runs/bert_{config.repr}_{config.dataset}_N{config.num_classes}_K{config.num_shot}"
        f"_Seed{config.random_seed}_HiddenDim{config.hidden_dim}_LR{config.learning_rate}_Dropout{config.dropout}"
    )

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

    # smiles_embedding_dim = 767, protein_embedding_dim = 640, vae_protein_embedding_dim = 100
    repr_to_input_dims = {"smiles_only": config.num_classes + 767,
                          "concat_after": config.num_classes + 767,
                          "concat_after_full": config.num_classes + 767,
                          "concat": config.num_classes + 767 + 640,
                          "concat_smiles_vaeprot": config.num_classes + 767 + 100}

    # Create BERT model configuration
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
        repr = config.repr
    )

    # Create model
    model = SmilesBertModel(model_config)
    model.to(device)

    # Create optimizer
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    times = []
    best_val_acc = 0
    for step in tqdm(range(config.train_steps)):
        ## Sample Batch
        t0 = time.time()
        i, l = next(train_loader)
        i, l = i.to(device), l.to(device)
        t1 = time.time()

        ## Train
        attention_mask = torch.ones((config.meta_batch_size, (config.num_shot+1)*config.num_classes))
        attention_mask = attention_mask.to(device)
        ls, _ = model(i.float(), l.float(), attention_mask)
        ls.backward()
        optim.step()
        optim.zero_grad()

        t2 = time.time()
        writer.add_scalar("Loss/train", ls, step)
        times.append([t1 - t0, t2 - t1])

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
            
            pred = torch.argmax(pred, axis=1)

            l = torch.argmax(l[:, -1, :, :], axis=1)

            acc = pred.eq(l).sum().item() / (
                config.meta_batch_size * config.num_classes
            )
            print("Val Accuracy", acc)
            writer.add_scalar("Accuracy/val", acc, step)

            if acc > best_val_acc:
                torch.save(model, f'model/{config.save}.pt')
                print("Saved model.")
                best_val_acc = acc

            times = np.array(times)
            print(
                f"Sample time {times[:, 0].mean()} Train time {times[:, 1].mean()}"
            )
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
    parser.add_argument("--repr", type=str, default="smiles_only")  # alternatively "smiles_only", "concat", "vaesmiles_only", "concat_after", "concat_after_full"
    parser.add_argument("--dataset", type=str, default="full")  # alternatively "full"
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--save", type=str)
    main(parser.parse_args())
