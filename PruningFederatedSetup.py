from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from typing import Dict, List, Tuple, Optional

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context, Parameters, FitRes, parameters_to_ndarrays

import torch.nn.utils.prune as prune

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")
disable_progress_bar()

NUM_CLIENTS = 50
NUM_PARTITIONS = 50
NUM_ROUNDS = 150
BATCH_SIZE = 32


# Choose what type of pruning, pruning location, and Pruning amount.
STRUCTURED = False
SERVER_PRUNING = False
CLIENT_PRUNING = False
PRUNE_AMOUNT = 0.2


def prune_model(model, amount=0.2):
    # Ensure the model is of the correct type
    if isinstance(model, torch.nn.Module):
        # Your pruning logic goes here
        if (not STRUCTURED):
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=amount)
                    if name == "classifier.1":
                        prune.l1_unstructured(module, name="bias", amount=0)  # Pruning the bias only here
                
                    prune.remove(module, "weight")
                    if name == "classifier.1":
                        prune.remove(module, "bias")
        else:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)  # Prune entire filters
                    prune.remove(module, "weight")

                elif isinstance(module, torch.nn.Linear):
                    prune.ln_structured(module, name="weight", amount=amount, n=2, dim=1)  # Prune entire neurons

                    if name == "classifier.1":
                        prune.l1_unstructured(module, name="bias", amount=0)

                    prune.remove(module, "weight")
                    if name == "classifier.1":
                        prune.remove(module, "bias")

                
        return model
    
    else:
        raise TypeError(f"Expected model to be of type 'nn.Module', but got {type(model)}.")


class FlowerClient(NumPyClient):
    """
    Federated Learning client implementation for Flower using PyTorch models and CIFAR-100 data.

    This client handles:
    - Fetching model parameters from the server
    - Training the model locally on its data partition
    - Evaluating model performance on local validation data

    Attributes:
    -----------
    pid : int
        Partition ID representing the unique client in the federated setup.
    net : torch.nn.Module
        The local neural network model used for training and evaluation.
    trainloader : DataLoader
        DataLoader for the local training dataset partition.
    valloader : DataLoader
        DataLoader for the local validation dataset partition.

    Methods:
    --------
    get_parameters(config):
        Returns the current local model parameters to the server.

    fit(parameters, config):
        Trains the local model using provided parameters and configuration,
        then returns the updated weights.

    evaluate(parameters, config):
        Evaluates the provided parameters on the local validation set,
        returning the loss and accuracy.
    """
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid  # partition ID of a client
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.pid}] get_parameters")
        if CLIENT_PRUNING:
            self.net = prune_model(self.net, amount=PRUNE_AMOUNT)
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.pid}, round {server_round}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=local_epochs)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.pid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

class AdaptiveFedAvg(FedAvg):
    """
    Custom implementation of the FedAvg strategy with weighted aggregation based on client sample sizes.

    This subclass of `flwr.server.strategy.FedAvg` overrides the `aggregate_fit` method
    to perform a more explicit and customizable weighted average of client model updates.

    The primary goal is to ensure that clients with more data influence the aggregated model more heavily.

    Methods:
    --------
    aggregate_fit(server_round, results, failures):
        Aggregates the updated model parameters received from clients using a weighted average
        based on the number of examples used in training.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[flwr.server.client_proxy.ClientProxy, flwr.common.FitRes]],
        failures: List,
    ) -> Tuple[Optional[Parameters], dict]:
        """
        Aggregate client model updates by computing a weighted average.

        Parameters:
        -----------
        server_round : int
            The current federated learning round.
        results : List[Tuple[ClientProxy, FitRes]]
            A list of tuples, each containing a client proxy and its training result.
        failures : List
            A list of client training failures (ignored in this implementation).

        Returns:
        --------
        Tuple[Optional[Parameters], dict]
            - Aggregated model parameters as a Flower `Parameters` object.
            - An empty dictionary for any additional metrics.
        """

        # Total samples used for weighting
        total_samples = sum(fit_res.num_examples for _, fit_res in results)

        # Weighted updates by number of client samples
        weighted_updates = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        if SERVER_PRUNING:
            aggregated = []

            for layers in zip(*[w[0] for w in weighted_updates]):
                weighted_sum = sum(layer * (num_samples / total_samples) for layer, num_samples in zip(layers, [w[1] for w in weighted_updates]))
                # Ensure that the result is a numpy array
                if np.isscalar(weighted_sum):
                    weighted_sum = np.array([weighted_sum])
                aggregated.append(weighted_sum)

            # print("Aggregated parameter shapes:", [param.shape for param in aggregated])
        

            model = get_model()
            set_parameters(model, aggregated)

            # for name, param in model.named_parameters():
            #     print(f"{name}: {param.shape}")

            model = prune_model(model, amount=PRUNE_AMOUNT)

            # for name, param in model.named_parameters():
            #     print(f"{name}: {param.shape}")
            
            pruned_parameters = get_parameters(model)

            # Fix scalars in pruned_parameters
            for i, param in enumerate(pruned_parameters):
                if param.shape == ():
                    pruned_parameters[i] = np.array([param], dtype=np.float32)

            # print("Pruned parameter shapes:", [param.shape for param in pruned_parameters])
            return ndarrays_to_parameters(pruned_parameters), {}
        
        else:
            # Aggregated parameters using weighted average
            aggregated = [
                sum(layer * (num_samples / total_samples) for layer, num_samples in zip(layers, [w[1] for w in weighted_updates]))
                for layers in zip(*[w[0] for w in weighted_updates])
            ]
        
    
            return ndarrays_to_parameters(aggregated), {}
    


class MetricsTracker:
    """
    Tracks and summarizes performance metrics across federated learning rounds.

    This class is used to monitor centralized evaluation metrics such as accuracy and loss,
    track training time, and provide insights into model convergence and stability.

    Attributes:
    -----------
    centralized_accuracy : List[float]
        A list of accuracy values collected after each server-side evaluation round.
    centralized_loss : List[float]
        A list of loss values collected after each server-side evaluation round.
    start_time : float
        Timestamp marking when training started (used for timing analysis).

    Methods:
    --------
    update(accuracy, loss):
        Append the latest evaluation accuracy and loss to their respective logs.

    report():
        Print a summary of training progress including best accuracy, corresponding loss,
        round of convergence, variance of accuracy over the last 10 rounds (stability),
        and total training duration.
    """
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

def get_model():
    """
    Initialize and return a MobileNetV2 model adapted for CIFAR-100 classification.

    This function constructs a MobileNetV2 architecture without loading default pretrained weights.
    It attempts to load pretrained ImageNet weights from a local checkpoint file. If the file is
    not found or an error occurs, the model continues with random initialization.

    The final classification layer is replaced to output 100 classes, as required for CIFAR-100.

    Returns:
    --------
    torch.nn.Module
        A MobileNetV2 model moved to the appropriate device (CPU or GPU).
    """
    
    from torchvision.models import MobileNet_V2_Weights

    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    model = mobilenet_v2(weights=weights)

    # model = mobilenet_v2(weights=None)
    # try:
    #     state_dict = torch.load(r"C:\Users\joe_w\.cache\torch\hub\checkpoints\mobilenet_v2-b0353104.pth", map_location=DEVICE)
    #     model.load_state_dict(state_dict)
    # except Exception as e:
    #     print(f"Could not load pretrained weights: {e}")

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
    return model.to(DEVICE)

def train(net, trainloader, epochs: int, verbose=False):
    """
    Train a neural network model on a local dataset using cross-entropy loss and Adam optimizer.

    This function performs supervised training over the specified number of epochs,
    using the provided DataLoader of image-label batches. It tracks cumulative loss and
    accuracy per epoch and optionally prints metrics for each epoch.

    Parameters:
    -----------
    net : torch.nn.Module
        The model to be trained.
    trainloader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    epochs : int
        Number of epochs to train the model for.
    verbose : bool, optional (default=False)
        If True, prints loss and accuracy at the end of each epoch.
    """

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"], batch["fine_label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """
    Evaluate a trained neural network model on a test dataset.

    This function performs inference on the entire test set without updating model weights.
    It calculates and returns the average cross-entropy loss and classification accuracy.

    Parameters:
    -----------
    net : torch.nn.Module
        The trained model to be evaluated.
    testloader : torch.utils.data.DataLoader
        DataLoader for the test dataset.

    Returns:
    --------
    loss : float
        Average cross-entropy loss over the test dataset.
    accuracy : float
        Classification accuracy as a float between 0 and 1.
    """
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"], batch["fine_label"]
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
    """
    Load and preprocess CIFAR-100 dataset partitions for a federated learning client.

    This function loads a specific data partition from the CIFAR-100 dataset using Flower's
    FederatedDataset utility. It splits the client's data into training and validation sets,
    applies standard image transformations (normalization to ImageNet stats), and loads the
    centralized test set shared by all clients.

    Parameters:
    -----------
    partition_id : int
        The ID of the client's data partition to load.
    num_partitions : int
        Total number of partitions the dataset has been split into.

    Returns:
    --------
    trainloader : torch.utils.data.DataLoader
        DataLoader for the client's local training data (80% of the partition).
    valloader : torch.utils.data.DataLoader
        DataLoader for the client's local validation data (20% of the partition).
    testloader : torch.utils.data.DataLoader
        DataLoader for the centralized test set used in server-side evaluation.
    """
    fds = FederatedDataset(dataset="cifar100", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
    ])

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader

def set_parameters(net, parameters: List[np.ndarray]):
    """
    Load a list of NumPy arrays into a PyTorch model's state_dict.

    This function takes a list of NumPy arrays representing model weights
    (typically received from the federated learning server), and updates
    the model's internal parameters accordingly.

    Parameters:
    -----------
    net : torch.nn.Module
        The PyTorch model whose parameters are to be updated.
    parameters : List[np.ndarray]
        A list of NumPy arrays corresponding to the model's state_dict values,
        in the same order as returned by `state_dict().values()`.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    """
    Extract the model's parameters as a list of NumPy arrays.

    This function retrieves the current state_dict from a PyTorch model,
    moves each tensor to the CPU (if needed), and converts them into
    NumPy arrays. It is typically used to send model weights to the
    federated learning server or aggregator.

    Parameters:
    -----------
    net : torch.nn.Module
        The PyTorch model from which to extract parameters.

    Returns:
    --------
    List[np.ndarray]
        A list of NumPy arrays representing the model's parameters.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Compute the weighted average accuracy across multiple clients.

    This function aggregates client-reported accuracy scores by weighting each one
    according to the number of examples used in training. It is typically used to
    compute a global accuracy metric in federated learning setups.

    Parameters:
    -----------
    metrics : List[Tuple[int, Metrics]]
        A list of tuples where each tuple contains:
        - The number of examples used by the client.
        - A dictionary of evaluation metrics reported by the client (must include "accuracy").

    Returns:
    --------
    Metrics : dict
        A dictionary containing the weighted average accuracy across all clients, with key "accuracy".
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """
    Construct and return the server-side configuration for a Flower federated learning simulation.

    This function defines the federated learning strategy and server behavior by configuring:
    - The aggregation strategy (AdaptiveFedAvg)
    - The number of training rounds
    - Evaluation and fit configuration functions
    - Minimum number of clients required for training and evaluation

    The configuration is wrapped in a `ServerAppComponents` object, which Flower uses to
    initialize the server for the simulation.

    Parameters:
    -----------
    context : flwr.common.Context
        Runtime context provided by Flower, which may contain run configuration or metadata.

    Returns:
    --------
    ServerAppComponents
        An object containing the configured strategy and server settings for the simulation.
    """

    # Create the FedAvg strategy
    strategy = AdaptiveFedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=evaluate,
        on_fit_config_fn=fit_config,
    )

    # Configure the server for 15 rounds of training
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    net = get_model()

    # Load data (CIFAR-100)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, _ = load_datasets(partition_id=partition_id, num_partitions=NUM_PARTITIONS)

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
    """
    Perform server-side evaluation of the global model using a centralized test set.

    This function is called by the Flower server after each federated round to evaluate
    the aggregated global model. It loads the test dataset, updates the model with the
    latest parameters, runs evaluation, and logs the results using a tracking utility.

    Parameters:
    -----------
    server_round : int
        The current federated learning round.
    parameters : NDArrays
        A list of NumPy arrays representing the latest global model parameters.
    config : Dict[str, Scalar]
        A dictionary of additional configuration values (unused in this function but required by Flower).

    Returns:
    --------
    Optional[Tuple[float, Dict[str, Scalar]]]
        A tuple containing:
        - The average loss over the test set (float)
        - A dictionary with the accuracy metric (key: "accuracy")
    """
    net = get_model()
    _, _, testloader = load_datasets(0, NUM_PARTITIONS)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader)
    # Update tracker
    tracker.update(accuracy, loss)
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}

def fit_config(server_round: int):
    """
    Generate a dynamic training configuration dictionary for each federated learning round.

    This function adjusts client-side training parameters based on the current server round:
    - The number of local epochs increases by 1 every 10 rounds (starting with 1 epoch at round 0).
    - The learning rate is set to 0.01 initially, and reduces to 0.005 after round 50.

    Parameters:
    ----------
    server_round : int
        The current federated learning round number (0-indexed).

    Returns:
    -------
    dict
        A configuration dictionary containing:
        - 'server_round': the current round number.
        - 'local_epochs': number of epochs each client should train locally this round.
        - 'learning_rate': the learning rate clients should use during training this round.
    """
    config = {
        "server_round": server_round,
        "local_epochs": 1 + server_round // 10,  # Increase epochs every 10 rounds
        "learning_rate": 0.01 if server_round < 50 else 0.005  # Adjust learning rate after 50 rounds
    }
    return config

# Trigger download once before the FL process
load_dataset("cifar100")

# Instantiate tracker
tracker = MetricsTracker()

# Create the ClientApp
client = ClientApp(client_fn=client_fn)

# Create an instance of the model and get the parameters
params = get_parameters(get_model())

# Create ServerApp
server = ServerApp(server_fn=server_fn)

# Specify the resources each of your clients need
# If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
backend_config = {"client_resources": None}
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_gpus": 1}}

# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)

# Print summary metrics
tracker.report()