import argparse
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from data_loader import DataGenerator
from tqdm import tqdm


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class BlackBoxLSTM(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim, input_dim, dropout_prob, repr):
        super(BlackBoxLSTM, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.input_dim = input_dim
        self.dropout_prob = dropout_prob
        self.repr = repr

        self.layer1 = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout_prob)
        if repr == "concat_after":
            hidden_dim += 100
        if repr == "concat_after_full":
            hidden_dim += 640
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        input_labels = input_labels.clone()
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

        output = self.layer1(input_images_and_labels.float())
        if self.repr == "concat_after":
            protein_embeds = protein_embeds.reshape((B, -1, 100))
            output = torch.concat((output[0], protein_embeds), axis=-1)
        elif self.repr == "concat_after_full":
            protein_embeds = protein_embeds.reshape((B, -1, 640))
            output = torch.concat((output[0], protein_embeds), axis=-1)
        else:
            output = output[0]
        predictions = self.layer2(self.dropout(output))[0]
        
        return predictions.reshape((B, K_1, N, N))

    def loss_function(self, preds, labels):
        query_preds = preds[:, -1, :, :]
        query_labels = labels[:, -1, :, :]
        loss = F.cross_entropy(query_preds, query_labels)
        return loss


def train_step(images, labels, model, optim, eval=False):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions.detach(), loss.detach()


def main(config):
    print(config)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    writer = SummaryWriter(
        f"runs/{config.repr}_{config.dataset}_N{config.num_classes}_K{config.num_shot}_Steps{config.train_steps}"
        f"_Seed{config.random_seed}_HiddenDim{config.hidden_dim}_LR{config.learning_rate}_Dropout{config.dropout}"
    )

    # Create Data Generator
    train_iterable = DataGenerator(
        data_json_path=f'data/{config.dataset}_train.json',
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
        data_json_path=f'data/{config.dataset}_test.json',
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
                          "concat": config.num_classes + 767 + 640,
                          "concat_after_full": config.num_classes + 767,
                          "concat_smiles_vaeprot": config.num_classes + 767 + 100}

    # Create model
    model = BlackBoxLSTM(
        config.num_classes,
        config.num_shot + 1,
        config.hidden_dim,
        repr_to_input_dims[config.repr],
        config.dropout,
        config.repr,
    )
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
        _, ls = train_step(i, l, model, optim)
        t2 = time.time()
        writer.add_scalar("Loss/train", ls, step)
        times.append([t1 - t0, t2 - t1])

        ## Evaluate
        if (step + 1) % config.eval_freq == 0:
            print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)
            i, l = next(test_loader)
            i, l = i.to(device), l.to(device)
            pred_logits, tls = train_step(i, l, model, optim, eval=True)
            # pred_logits ~ (B, K+1, N, N)
            # model sees K support examples for each of N classes and predicts on 1 query example for each of N classes (all shuffled ofc) -> (_, K+1, N, _)
            # here, LSTM outputs logits for each of N classes for each of the (K+1) * N support & query examples -> (_, K+1, N, N)
            # batch to leverage parallelism -> (B, K+1, N, N)
            print(
                "Train Loss:",
                ls.cpu().numpy(),
                "Test Loss:",
                tls.cpu().numpy(),
            )
            writer.add_scalar("Loss/test", tls, step)

            pred_logits = torch.reshape(
                pred_logits,
                [
                    -1,
                    config.num_shot + 1,
                    config.num_classes,
                    config.num_classes,
                ],
            )  # no change, already in correct shape
            pred_class = torch.argmax(pred_logits[:, -1, :, :], axis=2)
            # pred_logits[:, -1, :, :] selects logits for query example (not K support examples) over entire batch, shape ~ (B, N, N)
            # torch.argmax(..., axis=2) selects predicted class (with largest logit) for the N query examples (1 for each class), shape ~ (B, N)
            true_class = torch.argmax(l[:, -1, :, :], axis=2)  # selects ground-truth class for the N query examples
            acc = pred_class.eq(true_class).sum().item() / (
                config.meta_batch_size * config.num_classes
            )  # sums the number of matches between predicted and ground-truth class, divided by # of query examples in batch (B * N)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_shot", type=int, default=3)  # 10
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=10)  # 500
    parser.add_argument("--meta_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--train_steps", type=int, default=25000)
    parser.add_argument("--image_caching", type=bool, default=True)
    parser.add_argument("--repr", type=str, default="smiles_only")
    # "smiles_only", "concat", "vaesmiles_only", "concat_smiles_vaeprot", "concat_after"
    parser.add_argument("--dataset", type=str, default="dev")
    # "dev", "full"
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--save", type=str)
    main(parser.parse_args())
