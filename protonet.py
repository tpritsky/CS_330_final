"""Implementation of prototypical networks for Omniglot."""

import argparse
import os

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import tensorboard

from data_loader import DataGenerator

NUM_INPUT_CHANNELS = 1
NUM_HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
NUM_CONV_LAYERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SUMMARY_INTERVAL = 10
SAVE_INTERVAL = 100
PRINT_INTERVAL = 10
VAL_INTERVAL = PRINT_INTERVAL * 5
NUM_TEST_TASKS = 600


class ProtoNetNetwork(nn.Module):
    """Container for ProtoNet weights and image-to-latent computation."""

    def __init__(self, input_dim, hidden_dim, embedding_dim):
        """Inits ProtoNetNetwork.

        The network consists of four convolutional blocks, each comprising a
        convolution layer, a batch normalization layer, ReLU activation, and 2x2
        max pooling for downsampling. There is an additional flattening
        operation at the end.

        Note that unlike conventional use, batch normalization is always done
        with batch statistics, regardless of whether we are training or
        evaluating. This technically makes meta-learning transductive, as
        opposed to inductive.
        """
        super(ProtoNetNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.embedding_dim)

        self._layers = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU()
        )

        self.to(DEVICE)

    def forward(self, images):
        """Computes the latent representation of a batch of images.

        Args:
            images (Tensor): batch of Omniglot images
                shape (num_images, channels, height, width)

        Returns:
            a Tensor containing a batch of latent representations
                shape (num_images, latents)
        """
        return self._layers(images)


class ProtoNet:
    """Trains and assesses a prototypical network."""

    def __init__(self, learning_rate, log_dir, input_dim, hidden_dim, embedding_dim, num_support):
        """Inits ProtoNet.

        Args:
            learning_rate (float): learning rate for the Adam optimizer
            log_dir (str): path to logging directory
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.num_support = num_support

        self._network = ProtoNetNetwork(self.input_dim, self.hidden_dim, self.embedding_dim)
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
            task_batch (tuple[Tensor, Tensor, Tensor, Tensor]):
                batch of tasks from an Omniglot DataLoader

        Returns:
            a Tensor containing mean ProtoNet loss over the batch
                shape ()
            mean support set accuracy over the batch as a float
            mean query set accuracy over the batch as a float
        """
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []

        # In HW2, task_batch is a list of B=16 task tuples -> each task tuple contains 4 tensors
        # 1. images_support: shape (N=5*K=2, 1, H=28, W=28)
        # 2. labels_support: shape (N=5*K=2,) <- tensor of ints corresponding to label class
        # 3. images_query: shape (N=5 * 15, 1, H=28, W=28)
        # 4. labels_support: shape (N=5 * 15,) <- tensor of ints corresponding to label class

        inputs, labels = task_batch
        # inputs ~ (B, K+Q, N, input_dim)
        # labels ~ (B, K+Q, N, N)
        inputs = inputs.float()
        labels = torch.argmax(labels.float(), axis=-1)  # convert one-hot to int label -> (B, K+Q, N)

        B = inputs.shape[0]
        K = self.num_support
        Q = inputs.shape[1] - K
        N = inputs.shape[2]
        # D = inputs.shape[3]

        batch_support_inputs = inputs[:, :self.num_support, :, :]  # (B, K, N, input_dim)
        batch_query_inputs = inputs[:, self.num_support:, :, :]  # (B, Q, N, input_dim)
        batch_support_labels = labels[:, :self.num_support, :]  # (B, K, N)
        batch_query_labels = labels[:, self.num_support:, :]  # (B, Q, N)

        # TODO: compute prototypes across batch in one step
        """
        # 1. prototype embeddings
        batch_support_embeddings = self._network(batch_support_inputs)  # (B, K, N, embedding_dim)
        prototype_embeddings = torch.mean(batch_support_embeddings, dim=1)  # (B, N, embedding_dim)
        # then compute mean embedding (size embedding_dim) for each of the N classes
        # from the K embeddings for each class (reduce dim 1)
        # output has shape (B, N, embedding_dim)

        # 2. distance metric (squared euclidian distance)
        batch_query_embeddings = self._network(batch_query_inputs)  # (B, Q, N, embedding_dim)
        query_distances = torch.square(torch.cdist(batch_query_embeddings, prototype_embeddings))  # shape ~ (N * Q, N)
        support_distances = torch.square(
            torch.cdist(batch_support_embeddings, prototype_embeddings))  # shape ~ (N * K, N)
        # torch.cdist(x1, x2, ...) defaults to p=2 i.e. Euclidean distance, expects x1 ~ (B, P, M), x2 ~ (B, R, M)

        # 3. softmax of similarity score (-1 * distance)
        logits_query = F.softmax(-1 * query_distances, dim=1)  # shape (N * Q, N)
        logits_support = F.softmax(-1 * support_distances, dim=1)  # shape (N * K, N)
        # dim=1 to perform softmax across cols -> every slice along dim=1 (cols) sums to 1

        # 4. cross entropy loss and accuracy
        loss = F.cross_entropy(logits_query, batch_query_labels)
        accuracy_query = self._score(logits_query, batch_query_labels)
        accuracy_support = self._score(logits_support, batch_support_labels)
        """


        # TODO: compute prototypes by iterating along dim 0
        for i in range(B):
            images_support = batch_support_inputs[i].to(DEVICE)  # (K, N, embedding_dim)
            labels_support = batch_support_labels[i].to(DEVICE).flatten()  # (K*N,), CE loss & _score() expect 1D vector
            images_query = batch_query_inputs[i].to(DEVICE)  # (Q, N, embedding_dim)
            labels_query = batch_query_labels[i].to(DEVICE).flatten()  # (Q*N,), _score() expects 1D vector
            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
            # TODO: finish implementing this method.
            # For a given task, compute the prototypes and the protonet loss.
            # Use F.cross_entropy to compute classification losses.
            # Use util.score to compute accuracies.
            # Make sure to populate loss_batch, accuracy_support_batch, and
            # accuracy_query_batch.

            # Specifications for HW2
            # N=5 classes, K=5 support example per class, Q=15 query examples per class
            # default batch size B=16 (# tasks per outer loop update)

            # 1. prototype embeddings
            images_support_embeddings = self._network(images_support)  # shape ~ (K, N, embedding_dim)
            prototype_embeddings = torch.mean(images_support_embeddings, dim=0)  # shape ~ (N, embedding_dim)
            # for each of N classes, avg the K support embeddings to get prototype embedding

            # 2. distance metric (squared euclidian distance)
            images_query_embeddings = self._network(images_query)  # shape ~ (Q, N, embedding_dim) # TODO may need images_query.reshape(-1, embedding_dim).
            query_distances = torch.square(torch.cdist(images_query_embeddings.reshape(-1, self.embedding_dim), prototype_embeddings))  # shape ~ (N * Q, N)
            support_distances = torch.square(torch.cdist(images_support_embeddings.reshape(-1, self.embedding_dim), prototype_embeddings))  # shape ~ (N * K, N)
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

            # ********************************************************
            # ******************* YOUR CODE HERE *********************
            # ********************************************************
        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch)
        )

    def train(self, dataloader_train, dataloader_val, writer, num_training_steps):
        """Train the ProtoNet.

        Consumes dataloader_train to optimize weights of ProtoNetNetwork
        while periodically validating on dataloader_val, logging metrics, and
        saving checkpoints.

        Args:
            dataloader_train (DataLoader): loader for train tasks
            dataloader_val (DataLoader): loader for validation tasks
            writer (SummaryWriter): TensorBoard logger
        """
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
                writer.add_scalar('loss/train', loss.item(), i_step)
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
                    # for val_task_batch in dataloader_val: <- eval on 'full' validation set (args.meta_batch_size * 4) from dataloader_val in main())
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
                writer.add_scalar('loss/val', loss, i_step)
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

            if i_step % SAVE_INTERVAL == 0:
                self._save(i_step)

            if i_step >= num_training_steps:
                break  # stop after training for num_training_steps

    def test(self, dataloader_test):
        """Evaluate the ProtoNet on test tasks.

        Args:
            dataloader_test (DataLoader): loader for test tasks
        """
        accuracies = []
        for task_batch in dataloader_test:
            accuracies.append(self._step(task_batch)[2])
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        mean_95_confidence_interval = 1.96 * std / np.sqrt(NUM_TEST_TASKS)
        print(
            f'Accuracy over {NUM_TEST_TASKS} test tasks: '
            f'mean {mean:.3f}, '
            f'95% confidence interval {mean_95_confidence_interval:.3f}'
        )

    def load(self, checkpoint_step):
        """Loads a checkpoint.

        Args:
            checkpoint_step (int): iteration of checkpoint to load

        Raises:
            ValueError: if checkpoint for checkpoint_step is not found
        """
        target_path = (
            f'{os.path.join(self._log_dir, "state")}'
            f'{checkpoint_step}.pt'
        )
        if os.path.isfile(target_path):
            state = torch.load(target_path)
            self._network.load_state_dict(state['network_state_dict'])
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
            self._start_train_step = checkpoint_step + 1
            print(f'Loaded checkpoint iteration {checkpoint_step}.')
        else:
            raise ValueError(
                f'No checkpoint for iteration {checkpoint_step} found.'
            )

    def _save(self, checkpoint_step):
        """Saves network and optimizer state_dicts as a checkpoint.

        Args:
            checkpoint_step (int): iteration to label checkpoint with
        """
        torch.save(
            dict(network_state_dict=self._network.state_dict(),
                 optimizer_state_dict=self._optimizer.state_dict()),
            f'{os.path.join(self._log_dir, "state")}{checkpoint_step}.pt'
        )
        print('Saved checkpoint.')


def main(args):
    # Initialize Tensorboard logging
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'runs/protonet/{args.repr}.{args.dataset}.n:{args.num_classes}.k:{args.num_support}.' + \
                  f'q:{args.num_query}.lr:{args.learning_rate}.hd:{args.hidden_dim}.embed:{args.embedding_dim}.batch_size:{args.meta_batch_size}'
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
                        embedding_dim=args.embedding_dim,
                        num_support=args.num_support)

    if args.checkpoint_step > -1:
        protonet.load(args.checkpoint_step)
    else:
        print('Checkpoint loading skipped.')

    if not args.test:
        # num_training_steps = args.meta_batch_size * (args.num_train_iterations -
        #                                              args.checkpoint_step - 1)

        num_training_steps = args.meta_batch_size * args.num_train_iterations
        # number of tasks per outer-loop update * number of outer-loop updates to train for

        print(
            f'Training on tasks with composition '
            f'num_classes={args.num_classes}, '
            f'num_support={args.num_support}, '
            f'num_query={args.num_query}'
        )
        # dataloader_train = omniglot.get_omniglot_dataloader(
        #     'train',
        #     args.meta_batch_size,
        #     args.num_classes,
        #     args.num_support,
        #     args.num_query,
        #     num_training_steps
        # )
        # dataloader_val = omniglot.get_omniglot_dataloader(
        #     'val',
        #     args.meta_batch_size,
        #     args.num_classes,
        #     args.num_support,
        #     args.num_query,
        #     args.meta_batch_size * 4
        # )

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
            # dataloader_train,
            # dataloader_val,
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
        # dataloader_test = omniglot.get_omniglot_dataloader(
        #     'test',
        #     1,
        #     args.num_,
        #     args.num_support,
        #     args.num_query,
        #     NUM_TEST_TASKS
        # )

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
    parser.add_argument('--num_query', type=int, default=3,
                        help='number of query examples per class in a task')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate for the network')
    parser.add_argument('--meta_batch_size', type=int, default=16,
                        help='number of tasks per outer-loop update')
    parser.add_argument('--num_train_iterations', type=int, default=5000,
                        help='number of outer-loop updates to train for')
    parser.add_argument('--test', default=False, action='store_true',
                        help='train or test')
    parser.add_argument('--checkpoint_step', type=int, default=-1,
                        help=('checkpoint iteration to load for resuming '
                              'training, or for evaluation (-1 is ignored)'))

    # new CLI arguments
    parser.add_argument('--repr', type=str, default='concat_smiles_vaeprot',
                        help='representation of input proteins and ligands')
    parser.add_argument('--dataset', type=str, default='dev',
                        help='representation of input proteins and ligands')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='hidden dimension of ProtoNet MLP')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='prototype embedding dimension')
    parser.add_argument("--num_workers", type=int, default=4)


    main_args = parser.parse_args()
    main(main_args)
