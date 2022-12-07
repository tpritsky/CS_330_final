import argparse
import os
import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard

from data_loader import DataGenerator

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5


class ProtoNetNetwork(nn.Module):
    """Container for ProtoNet weights and input-to-latent computation."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """Inits ProtoNetNetwork
        ProtoNetNetwork is a a 2 layer MLP that projects input vectors into the latent
        (i.e. prototype embedding) space for nearest neighbors classification. 

        Args:
            input_dim (int): dimension of input embeddings
            hidden_dim (int): dimension of MLP hidden layer
            latent_dim (int): dimension of latent (i.e. prototype embedding) space
        """
        super(ProtoNetNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)

        self._layers = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU()
        )

        self.to(DEVICE)

    def forward(self, inputs):
        """Computes the latent representation of a batch of inputs.

        Args:
            inputs (Tensor): batch of input vectors
                shape (K, N, input_dim)

        Returns:
            a Tensor containing a batch of latent representations
                shape (K, N, latent_dim)
        """
        return self._layers(inputs)


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, learning_rate, log_dir, input_dim, hidden_dim, latent_dim, num_support, save_name):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
            input_dim (int): dimension of input embeddings
            hidden_dim (int): dimension of MLP hidden layer
            latent_dim (int): dimension of latent (i.e. prototype embedding) space
            num_support (int): K or number of support examples
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_support = num_support
        self.save_name = save_name

        self._network = ProtoNetNetwork(self.input_dim, self.hidden_dim, self.latent_dim)
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=learning_rate
        )
        self._log_dir = log_dir
        os.makedirs(self._log_dir, exist_ok=True)

        self._start_train_step = 0

    def _score(self, logits, labels):
        """Returns the mean accuracy of a model's predictions on a set of examples.

        Args:
            logits (torch.Tensor): model predicted logits
                shape (examples, classes)
            labels (torch.Tensor): classification labels from 0 to num_classes - 1
                shape (examples,)
        """

        assert logits.dim() == 2
        assert labels.dim() == 1
        assert logits.shape[0] == labels.shape[0]
        y = torch.argmax(logits, dim=-1) == labels
        y = y.type(torch.float)
        return torch.mean(y).item()

    def _step(self, task_batch):
        """Computes ProtoNet mean loss (and accuracy) on a batch of tasks.

        Args:
            task_batch (tuple[Tensor, Tensor]):
                tuple of (inputs, labels) from dataloader
                    inputs ~ (B, K+Q, N, input_dim)
                    labels ~ (B, K+Q, N, N)

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []

        inputs, labels = task_batch
        # inputs ~ (B, K+Q, N, input_dim)
        # labels ~ (B, K+Q, N, N)
        inputs = inputs.float()
        labels = torch.argmax(labels.float(), axis=-1)  # convert one-hot to int label -> (B, K+Q, N)

        B = inputs.shape[0]

        batch_support_inputs = inputs[:, :self.num_support, :, :]  # (B, K, N, input_dim)
        batch_query_inputs = inputs[:, self.num_support:, :, :]  # (B, Q, N, input_dim)
        batch_support_labels = labels[:, :self.num_support, :]  # (B, K, N)
        batch_query_labels = labels[:, self.num_support:, :]  # (B, Q, N)

        # Compute prototypes by iterating along dim 0
        for i in range(B):
            inputs_support = batch_support_inputs[i].to(DEVICE)  # (K, N, embedding_dim)
            labels_support = batch_support_labels[i].to(DEVICE).flatten()  # (K*N,), CE loss & _score() expect 1D vector
            inputs_query = batch_query_inputs[i].to(DEVICE)  # (Q, N, embedding_dim)
            labels_query = batch_query_labels[i].to(DEVICE).flatten()  # (Q*N,), _score() expects 1D vector
            
            # For a given task, compute the prototypes and the protonet loss.
            # Use F.cross_entropy to compute classification losses.
            # Use _score() to compute accuracies.
            # Populate loss_batch, accuracy_support_batch, accuracy_query_batch.

            # 1. prototype embeddings
            inputs_support_embeddings = self._network(inputs_support)  # shape ~ (K, N, latent_dim)
            prototype_embeddings = torch.mean(inputs_support_embeddings, dim=0)  # shape ~ (N, latent_dim)
            # for each of N classes, avg the K support embeddings to get prototype embedding

            # 2. distance metric (squared euclidian distance)
            inputs_query_embeddings = self._network(inputs_query)  # shape ~ (Q, N, latent_dim)
            query_distances = torch.square(torch.cdist(inputs_query_embeddings.reshape(-1, self.latent_dim), prototype_embeddings))  # shape ~ (N * Q, N)
            support_distances = torch.square(torch.cdist(inputs_support_embeddings.reshape(-1, self.latent_dim), prototype_embeddings))  # shape ~ (N * K, N)
            # torch.cdist(...) defaults to p=2 i.e. Euclidean distance, expects x1 ~ (B, P, M), x2 ~ (B, R, M)

            # 3. softmax of similarity score (-1 * distance)
            logits_query = F.softmax(-1 * query_distances, dim=1)  # shape (N * Q, N)
            logits_support = F.softmax(-1 * support_distances, dim=1)  # shape (N * K, N)
            # dim=1 to perform softmax across cols -> every slice along dim=1 (cols) sums to 1

            # 4. cross entropy loss and accuracy
            loss = F.cross_entropy(logits_query, labels_query)
            accuracy_query = self._score(logits_query, labels_query)
            accuracy_support = self._score(logits_support, labels_support)

            loss_batch.append(loss)
            accuracy_query_batch.append(accuracy_query)
            accuracy_support_batch.append(accuracy_support)

        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch)
        )

    def train(self, dataloader_train, dataloader_val, writer, num_training_steps):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        Periodically validate on dataloader_val, logging metrics, and
        save checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
            num_training_steps (int): number of steps to train for
        """
        best_val_query_acc = 0

        print(f'Starting training at iteration {self._start_train_step}.')
        for i_step, task_batch in enumerate(
                dataloader_train,
                start=self._start_train_step
        ):
            self._optimizer.zero_grad()
            loss, accuracy_support, accuracy_query = self._step(task_batch)
            loss.backward()
            self._optimizer.step()

            if i_step % PRINT_INTERVAL == 0:
                print(
                    f'Iteration {i_step}: '
                    f'loss: {loss.item():.3f}, '
                    f'support accuracy: {accuracy_support.item():.3f}, '
                    f'query accuracy: {accuracy_query.item():.3f}'
                )
                writer.add_scalar('Loss/train', loss.item(), i_step)
                writer.add_scalar(
                    'train_accuracy/support',
                    accuracy_support.item(),
                    i_step
                )
                writer.add_scalar(
                    'train_accuracy/query',
                    accuracy_query.item(),
                    i_step
                )

            if i_step % VAL_INTERVAL == 0:
                with torch.no_grad():
                    losses, accuracies_support, accuracies_query = [], [], []
                    val_task_batch = next(dataloader_val)  # eval on 1 batch from validation set // MANN
                    loss, accuracy_support, accuracy_query = (
                        self._step(val_task_batch)
                    )
                    losses.append(loss.item())
                    accuracies_support.append(accuracy_support)
                    accuracies_query.append(accuracy_query)

                    loss = np.mean(losses)
                    accuracy_support = np.mean(accuracies_support)
                    accuracy_query = np.mean(accuracies_query)
                print(
                    f'Validation: '
                    f'loss: {loss:.3f}, '
                    f'support accuracy: {accuracy_support:.3f}, '
                    f'query accuracy: {accuracy_query:.3f}'
                )
                writer.add_scalar('Loss/val', loss, i_step)
                writer.add_scalar(
                    'val_accuracy/support',
                    accuracy_support,
                    i_step
                )
                writer.add_scalar(
                    'val_accuracy/query',
                    accuracy_query,
                    i_step
                )
                # save model with best val query accuracy
                if accuracy_query > best_val_query_acc:
                    torch.save(self._network, f'model/{self.save_name}.pt')

            if i_step >= num_training_steps:
                break  # stop after training for num_training_steps

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        NUM_TEST_TASKS = 1000
        for _ in tqdm(range(NUM_TEST_TASKS)):
            task_batch = next(dataloader_test)
            task_batch = task_batch.to(DEVICE)
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, model_checkpoint):
        """Loads a checkpoint.

        Args:
            model_checkpoint (str): path of model checkpoint file to load

        Raises:
            ValueError: if file is not found
        """
        if os.path.isfile(model_checkpoint):
            self._network = torch.load(model_checkpoint).to(DEVICE)
        else:
            raise ValueError(
                f'No file {model_checkpoint} found.'
            )

    def _save(self, model_checkpoint_path):
        """Saves model state as a checkpoint.

        Args:
            model_checkpoint_path (str): file path for model checkpoint file to save
        """
        torch.save(self._network, f'{model_checkpoint_path}.pt')
        print('Saved checkpoint.')


def main(args):
    # Initialize Tensorboard logging
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'runs/protonet/{args.repr}.n:{args.num_classes}.k:{args.num_support}.' + \
                  f'q:{args.num_query}.lr:{args.learning_rate}.hd:{args.hidden_dim}.embed:{args.latent_dim}.' + \
                  f'batch_size:{args.meta_batch_size}.train_iter:{args.num_train_iterations}'
    print(f'log_dir: {log_dir}')
    writer = tensorboard.SummaryWriter(log_dir=log_dir)

    # Initialize model
    repr_to_input_dims = {"smiles_only": 767,
                          "concat": 767 + 640,
                          "concat_smiles_vaeprot": 767 + 100}
    # smiles_embedding_dim = 767, protein_embedding_dim = 640, vae_protein_embedding_dim = 100

    protonet = ProtoNet(learning_rate=args.learning_rate,
                        log_dir=log_dir,
                        input_dim=repr_to_input_dims[args.repr],
                        hidden_dim=args.hidden_dim,
                        latent_dim=args.latent_dim,
                        num_support=args.num_support,
                        save_name=args.save)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        num_training_steps = args.meta_batch_size * args.num_train_iterations
        # number of tasks per outer-loop update * number of outer-loop updates to train for

        print(
            f'Training on tasks with composition '
            f'num_classes={args.num_classes}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )

        train_iterable = DataGenerator(
            data_json_path=f'data/train.json',
            k=args.num_support,
            q=args.num_query,
            repr=args.repr,
        )
        train_loader = iter(
            torch.utils.data.DataLoader(
                train_iterable,
                batch_size=args.meta_batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )
        val_iterable = DataGenerator(
            data_json_path=f'data/val.json',
            k=args.num_support,
            q=args.num_query,
            repr=args.repr,
        )
        val_loader = iter(
            torch.utils.data.DataLoader(
                val_iterable,
                batch_size=args.meta_batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )
        protonet.train(
            train_loader,
            val_loader,
            writer,
            num_training_steps
        )
    else:  # args.test == True
        print(
            f'Testing on tasks with composition '
            f'num_classes={args.num_classes}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )

        test_iterable = DataGenerator(
            data_json_path=f'data/test.json',
            k=args.num_support,
            repr=args.repr,
        )
        test_loader = iter(
            torch.utils.data.DataLoader(
                test_iterable,
                batch_size=args.meta_batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        )
        protonet.test(test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a ProtoNet!')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory to save to or load from')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes in a task')
    parser.add_argument('--num_support', type=int, default=5,
                        help='number of support examples per class in a task')
    parser.add_argument('--num_query', type=int, default=15,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='learning rate for the network')
    parser.add_argument('--meta_batch_size', type=int, default=100,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=500,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))
    parser.add_argument('--repr', type=str, default='smiles_only',  # "smiles_only", "concat", "concat_smiles_vaeprot"
                        help='representation of input proteins and ligands')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden dimension of ProtoNet MLP')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='latent (i.e. prototype embedding) dimension')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save", type=str)

    main_args = parser.parse_args()
    main(main_args)
