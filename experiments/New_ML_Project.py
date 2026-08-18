from collections import OrderedDict
from typing import List, Tuple

import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import CIFAR100
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Subset
import warnings
from datasets import load_dataset
from datasets import config
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad, FedProx, FedAdam
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context, Parameters, FitRes, parameters_to_ndarrays

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

NUM_CLIENTS = 50
NUM_PARTITIONS = 50
NUM_ROUNDS = 100
BATCH_SIZE = 32

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR100ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.model.maxpool = nn.Identity()  # CIFAR images are small; skip max pooling
        self.model.fc = nn.Linear(self.model.fc.in_features, 100)

    def forward(self, x):
        return self.model(x)

class FlowerClient(NumPyClient):
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid  # partition ID of a client
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.pid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.pid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        global_weights = [param.detach().clone() for param in self.net.parameters()]
        train(self.net, self.trainloader, epochs=local_epochs, global_weights=global_weights, verbose=False)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.pid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

class MetricsTracker:
    def __init__(self):
        self.centralized_accuracy = []
        self.centralized_loss = []
        self.start_time = time.time()

    def update(self, accuracy, loss):
        self.centralized_accuracy.append(accuracy)
        self.centralized_loss.append(loss)

    def report(self):
        end_time = time.time()
        processing_time = end_time - self.start_time
        best_accuracy = max(self.centralized_accuracy)
        best_round = self.centralized_accuracy.index(best_accuracy) + 1  # Rounds are 1-indexed

        stability = float(np.var(self.centralized_accuracy[-10:])) if len(self.centralized_accuracy) >= 10 else 0

        print("\n===== Federated Learning Summary =====")
        print(f"Best Centralized Accuracy: {best_accuracy:.4f}")
        print(f"Corresponding Centralized Loss: {self.centralized_loss[best_round - 1]:.4f}")
        print(f"Convergence Speed (Round): {best_round}")
        print(f"Stability (variance last 10 rounds): {stability:.6f}")
        print(f"Processing Time (s): {processing_time:.2f}")

class AdaptiveFedAvg(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], dict]:

        # Total samples used for weighting
        total_samples = sum(fit_res.num_examples for _, fit_res in results)

        # Weighted updates by number of client samples
        weighted_updates = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        # Aggregated parameters using weighted average
        aggregated = [
            sum(layer * (num_samples / total_samples) for layer, num_samples in zip(layers, [w[1] for w in weighted_updates]))
            for layers in zip(*[w[0] for w in weighted_updates])
        ]

        return ndarrays_to_parameters(aggregated), {}

def prepare_partitioned_cifar100(data_root="./data", num_clients=50, seed=42):
    if os.path.exists(f"{data_root}/partitions/train_idx_0.npy"):
        print("Partitions already exist. Skipping split.")
        return
    train_data = CIFAR100(root=data_root, train=True, download=True)
    X = np.arange(len(train_data))
    y = np.array(train_data.targets)

    skf = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
    indices = list(skf.split(X, y))

    os.makedirs(f"{data_root}/partitions", exist_ok=True)
    for i, (train_idx, val_idx) in enumerate(indices):
        np.save(f"{data_root}/partitions/train_idx_{i}.npy", train_idx)
        np.save(f"{data_root}/partitions/val_idx_{i}.npy", val_idx)

def train(net, trainloader, epochs: int, global_weights=None, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=5e-4)
    net.train()

    if verbose:
        print(f"[Client] Starting training loop with {len(trainloader)} batches...")

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        if verbose:
            print(f"[Client] Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(trainloader):
            try:
                t0 = time.time()

                # Load batch
                images, labels = batch

                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                # Forward pass
                t1 = time.time()
                outputs = net(images)
                if verbose:
                    print(f"[Client] Batch {batch_idx+1}: forward pass in {time.time() - t1:.3f}s")

                # Loss
                loss = criterion(outputs, labels)

                # FedProx (optional)
                if global_weights is not None:
                    mu = 0.1
                    proximal_term = 0.0
                    for w, w0 in zip(net.parameters(), global_weights):
                        proximal_term += ((w - w0.to(DEVICE)) ** 2).sum()
                    loss += (mu / 2) * proximal_term

                # Backward pass
                t2 = time.time()
                loss.backward()
                if verbose:
                    print(f"[Client] Batch {batch_idx+1}: backward pass in {time.time() - t2:.3f}s")

                # Optimizer step
                t3 = time.time()
                optimizer.step()
                if verbose:
                    print(f"[Client] Batch {batch_idx+1}: optimizer step in {time.time() - t3:.3f}s")

                # Metrics
                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            except Exception as e:
                print(f"[Client] ERROR in batch {batch_idx+1}: {e}")
                return

        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total

        print(f"[Client] Epoch {epoch+1} complete: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

        if verbose:
            print(f"Verbose: Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    print("[Client] Finished training.")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def load_datasets(partition_id: int, num_partitions: int):
    data_root = "./data"
    
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    # Load global CIFAR100
    full_trainset = CIFAR100(root=data_root, train=True, transform=train_transforms)
    testset = CIFAR100(root=data_root, train=False, transform=test_transforms)

    # Load pre-split indices
    train_idx = np.load(f"{data_root}/partitions/train_idx_{partition_id}.npy")
    val_idx = np.load(f"{data_root}/partitions/val_idx_{partition_id}.npy")

    train_subset = Subset(full_trainset, train_idx)
    val_subset = Subset(full_trainset, val_idx)

    trainloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valloader = DataLoader(val_subset, batch_size=BATCH_SIZE, num_workers=4)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, num_workers=4)

    return trainloader, valloader, testloader

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in net.state_dict().values()]

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that configure the ServerApp behavior.

    Initializes the federated learning server using AdaptiveFedAvg strategy
    with initial parameters from a CIFAR100-specific ResNet18 model.

    Parameters:
    -----------
    context : Context
        Flower server context used for additional configuration.

    Returns:
    --------
    ServerAppComponents
        Configured AdaptiveFedAvg strategy and server configurations.
    """
    net = CIFAR100ResNet18()
    params = get_parameters(net)

    strategy = AdaptiveFedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=evaluate,
        on_fit_config_fn=lambda rnd: fit_config(rnd, use_cosine_schedule=True)  # set True to enable annealing
    )

    config = ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=600)
    return ServerAppComponents(strategy=strategy, config=config)


def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = CIFAR100ResNet18().to(DEVICE)

    # Load data (CIFAR-100)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, _ = load_datasets(partition_id=partition_id, num_partitions=NUM_PARTITIONS)
    print(f"[Client {partition_id}] Loaded {len(trainloader.dataset)} train samples and {len(valloader.dataset)} val samples")

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()

# The `evaluate` function will be called by Flower after every round
def evaluate(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    net = CIFAR100ResNet18().to(DEVICE)
    _, _, testloader = load_datasets(0, NUM_PARTITIONS)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader)

    # Update tracker
    tracker.update(accuracy, loss)

    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

def fit_config(server_round: int, use_cosine_schedule: bool = False) -> dict:
    """
    Generate training config:
    - Cosine Annealing + Warm Restarts if enabled.
    - Otherwise, use fixed learning rate.
    """

    max_epochs = 8
    local_epochs = min(1 + server_round // 15, max_epochs)

    if use_cosine_schedule:
        # Cosine Annealing with Warm Restarts
        T_i = 30  # Restart every 30 rounds
        initial_lr = 0.01
        final_lr = 0.001

        cycle_round = server_round % T_i
        lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + math.cos(math.pi * cycle_round / T_i))
    else:
        lr = 0.001  # Stable learning rate for Adam

    return {
        "server_round": server_round,
        "local_epochs": local_epochs,
        "learning_rate": lr
    }

prepare_partitioned_cifar100(num_clients=NUM_PARTITIONS)

# Instantiate tracker
tracker = MetricsTracker()

# Create the ClientApp
client = ClientApp(client_fn=client_fn)

# Create an instance of the model and get the parameters
net = CIFAR100ResNet18()
params = get_parameters(net)
initial_parameters = ndarrays_to_parameters(params)

# Create ServerApp
server = ServerApp(server_fn=server_fn)

# Specify the resources each of your clients need
# If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1}}

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )

    # Print summary metrics
    tracker.report()