from collections import OrderedDict
from typing import List, Tuple

import json

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import warnings
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from datasets import get_dataset_config_names
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import datasets as tv_datasets, transforms
from torchvision.datasets import SVHN
from typing import Dict, List, Tuple, Optional

import flwr
from flwr.client import Client, NumPyClient
from flwr.server.server_config import ServerConfig
from flwr.server.server_app import ServerApp
from flwr.server.serverapp_components import ServerAppComponents
from flwr.simulation import start_simulation
from flwr.common import Metrics, Context
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr_datasets.partitioner import IidPartitioner
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context, Parameters, FitRes, parameters_to_ndarrays, FitIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

import torch.nn.utils.prune as prune

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set environment variable for Flower backend
os.environ["FLWR_BACKEND"] = "multiprocessing"

# Define the device-getter function - full GPU usage for Lambda Labs
def get_device(index=0):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

# Use the device
DEVICE = get_device()  # Add this line
device = DEVICE
print(f"Training on {device}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

# Assuming disable_progress_bar is already defined
#disable_progress_bar()

# Config values - optimized for Lambda Labs
NUM_CLIENTS = 25
NUM_PARTITIONS = 25
NUM_ROUNDS = 10
BATCH_SIZE = 32  # Increased batch size for faster training

global_strategy = None
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
            print("NOTSTRUCTURED")
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=amount)
                    if name == "classifier.1":
                        prune.l1_unstructured(module, name="bias", amount=0)  # Pruning the bias only here
                
                    prune.remove(module, "weight")
                    if name == "classifier.1":
                        prune.remove(module, "bias")
        else:
            print("STRUCTURED")
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
        self.device = get_device()
        self.net = net.to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        print(f"[Client {self.pid}] Using device: {self.device}")

    def get_parameters(self, config):
        print(f"[Client {self.pid}] get_parameters")
        if CLIENT_PRUNING:
            self.net = prune_model(self.net, amount=PRUNE_AMOUNT)
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]

        # Use values provided by the config
        print(f"[Client {self.pid}, round {server_round}] fit, config: {config}")
        try:
            # Use self.net (not net)
            set_parameters(self.net, parameters)
        except RuntimeError as e:
            print(f"Skipping evaluation due to bad parameters: {e}")
            return None

        device = get_device()
        train(self.net, self.trainloader, epochs=local_epochs, device=device, verbose=True)

        # Return updated parameters
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.pid}] evaluate, config: {config}")
        if not parameters:
            print("Empty parameters, skipping evaluation.")
            return None
        set_parameters(self.net, parameters)
        loss, accuracy, precision, recall, f1 = test(self.net, self.valloader)
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

        # Aggregated parameters using weighted average
        aggregated = [
            sum(layer * (num_samples / total_samples) for layer, num_samples in zip(layers, [w[1] for w in weighted_updates]))
            for layers in zip(*[w[0] for w in weighted_updates])
        ]

        return ndarrays_to_parameters(aggregated), {}

class TrustFedAvg(FedAvg):
    def __init__(self, *args, warmup_rounds=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_scores = {}
        self.warmup_rounds = warmup_rounds

#     def configure_fit(
#     self, server_round: int, parameters: Parameters, client_manager: ClientManager
# )       -> List[Tuple[ClientProxy, FitIns]]:
        
#         if not SERVER_PRUNING:
#             print("Using default FedAvg configuration (no pruning)...")
#             return super().configure_fit(server_round, parameters, client_manager)
        
#          # Convert parameters to a PyTorch model
#         model = self.parameters_to_model(parameters)

#         # Apply pruning before sending the model to clients
#         pruned_model = self.prune_model(model, amount=0.2)

#         # Convert the pruned model back to parameters
#         pruned_parameters = self.model_to_parameters(pruned_model)

#         # Generate fit instructions with pruned parameters
#         fit_ins = FitIns(parameters=pruned_parameters, config={})

#         # Select clients
#         clients = (client_manager.all().values())

#         # Return client proxies along with the fit instructions
#         return [(client, fit_ins) for client in clients]

    # def parameters_to_model(self, parameters: Parameters) -> torch.nn.Module:
    #     """Convert Flower parameters to a PyTorch model."""
    #     model = get_model()  # Use the provided model function
    #     model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters_to_ndarrays(parameters))})
    #     return model

    # def model_to_parameters(self, model: torch.nn.Module) -> Parameters:
    #     """Convert a PyTorch model to Flower parameters."""
    #     ndarray_list = [v.cpu().numpy() for v in model.state_dict().values()]
    #     return ndarrays_to_parameters(ndarray_list)
    
    def aggregate_fit(self, server_round, results, failures):
        print("AGGREGATE FIT")
        total_weight = 0
        weighted_updates = []

        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            acc = fit_res.metrics.get("accuracy", 0.5)  # fallback if no accuracy
            samples = fit_res.num_examples

            # Update trust score
            # During warmup: default trust = 1.0
            if server_round < self.warmup_rounds:
                trust = 1.0
            else:
                prev = self.trust_scores.get(cid, 1.0)
                trust = 0.7 * prev + 0.3 * acc

            self.trust_scores[cid] = trust

            weight = trust * samples
            weighted_updates.append((parameters_to_ndarrays(fit_res.parameters), weight))
            total_weight += weight

        aggregated = [
            sum(layer * (weight / total_weight) for layer, weight in zip(layers, [w[1] for w in weighted_updates]))
            for layers in zip(*[w[0] for w in weighted_updates])
        ]

        if not hasattr(self, "trust_log"):
            self.trust_log = []

        avg_trust = np.mean(list(self.trust_scores.values()))
        self.trust_log.append(avg_trust)

        if not hasattr(self, "client_trust_log"):
            self.client_trust_log = {}

        for cid in ["client_0", "client_1", "client_2"]:
            if cid in self.trust_scores:
                self.client_trust_log.setdefault(cid, []).append(self.trust_scores[cid])

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
        self.centralized_precision = []
        self.centralized_recall = []
        self.centralized_f1 = []
        self.start_time = time.time()

    def update(self, accuracy, loss, precision=None, recall=None, f1=None):
        self.centralized_accuracy.append(accuracy)
        self.centralized_loss.append(loss)
        if precision is not None:
            self.centralized_precision.append(precision)
        if recall is not None:
            self.centralized_recall.append(recall)
        if f1 is not None:
            self.centralized_f1.append(f1)

    def report(self):
        end_time = time.time()
        processing_time = end_time - self.start_time
        best_accuracy = max(self.centralized_accuracy)
        best_round = self.centralized_accuracy.index(best_accuracy)

        stability = (
            float(np.var(self.centralized_accuracy[-10:]))
            if len(self.centralized_accuracy) >= 10
            else 0
        )

        print("\n===== Federated Learning Summary =====")
        print(f"Best Centralized Accuracy: {best_accuracy:.4f}")
        print(f"Corresponding Centralized Loss: {self.centralized_loss[best_round]:.4f}")
        if self.centralized_precision and len(self.centralized_precision) > best_round:
            print(f"Precision: {self.centralized_precision[best_round]:.4f}")
        if self.centralized_recall and len(self.centralized_recall) > best_round:
            print(f"Recall: {self.centralized_recall[best_round]:.4f}")
        if self.centralized_f1 and len(self.centralized_f1) > best_round:
            print(f"F1 Score: {self.centralized_f1[best_round]:.4f}")
        print(f"Convergence Speed (Round): {best_round + 1}")
        print(f"Stability (variance last 10 rounds): {stability:.6f}")
        print(f"Processing Time (s): {processing_time:.2f}")

def get_model():
    """
    Initialize and return an EfficientNet-B0 model adapted for CIFAR-100 classification.

    This function constructs an EfficientNet-B0 architecture with pretrained weights.
    The final classification layer is replaced to output 100 classes, as required for CIFAR-100.

    Returns:
    --------
    torch.nn.Module
        An EfficientNet-B0 model moved to the appropriate device (CPU or GPU).
    """
    # Initialize EfficientNet-B0 with pretrained weights
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 10)
    return model

def train(net, trainloader, epochs: int, device, verbose=False):
    """
    Train a neural network model on a local dataset using cross-entropy loss and Adam optimizer.

    Parameters:
    -----------
    net : torch.nn.Module
        The model to be trained.
    trainloader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    epochs : int
        Number of epochs to train the model for.
    device : torch.device
        The device to run training on (e.g., cpu or cuda).
    verbose : bool, optional (default=False)
        If True, prints loss and accuracy at the end of each epoch.
    """

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # Learning rate for EfficientNet
    net.to(device)
    net.train()

    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch
            labels = labels.long()
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

from sklearn.metrics import classification_report

def test(net, testloader):
    """
    Evaluate a trained neural network model on a test dataset and compute performance metrics.

    This function performs inference on the entire test set without updating model weights.
    It calculates and returns the average cross-entropy loss, classification accuracy,
    and weighted precision, recall, and F1-score.

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
    precision : float
        Weighted precision score over all classes.
    recall : float
        Weighted recall score over all classes.
    f1 : float
        Weighted F1 score over all classes.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    all_preds, all_labels = [], []

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            if isinstance(batch, dict):
                images, labels = batch["img"], batch["fine_label"]
            else:
                images, labels = batch
            device = next(net.parameters()).device
            labels = labels.long()
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(testloader))
    accuracy = correct / total

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    return avg_loss, accuracy, precision, recall, f1


def load_datasets(partition_id: int, num_partitions: int):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Optional upscale for EfficientNet
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download and load SVHN
    full_train = SVHN(root="./data", split="train", download=True, transform=transform)
    full_test = SVHN(root="./data", split="test", download=True, transform=transform)

    # Partition training set
    partition_size = len(full_train) // num_partitions
    lengths = [partition_size] * num_partitions
    for i in range(len(full_train) % num_partitions):
        lengths[i] += 1
    train_partitions = random_split(full_train, lengths, generator=torch.Generator().manual_seed(123))
    client_dataset = train_partitions[partition_id]

    # Split into train/val
    val_size = int(0.2 * len(client_dataset))
    train_size = len(client_dataset) - val_size
    train_split, val_split = random_split(client_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(321))

    trainloader = DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valloader = DataLoader(val_split, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    testloader = DataLoader(full_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

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
    # Check if parameters are valid and non-empty
    if not parameters or len(parameters) == 0:
        print("Warning: Empty parameters list, skipping parameter update")
        return
    # Get the expected keys in the correct order
    state_dict_keys = list(net.state_dict().keys())
    # Validate parameter count matches
    if len(parameters) != len(state_dict_keys):
        print(f"Warning: Parameter count mismatch. Expected {len(state_dict_keys)}, got {len(parameters)}")
        # For safety, return without loading mismatched parameters
        return
    # Create state dict with correct key-value pairs
    params_dict = zip(state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # Load parameters with error reporting
    try:
        net.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"Error loading state dict: {str(e)}")

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

    # Create a properly initialized model and extract parameters
    model = get_model()
    init_params = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create the strategy - use a higher fraction for Lambda Labs
    global global_strategy
    global_strategy = TrustFedAvg(
        warmup_rounds=15,  # <-- adjust as needed
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(init_params),
        evaluate_fn=evaluate,
        on_fit_config_fn=fit_config,
    )

    # Configure the server for the specified number of training rounds
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=global_strategy, config=config)

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    net = get_model()

    # Debug logging to understand the context structure
    print(f"Context node_config: {context.node_config}")
    
    # Try to get partition_id, with fallback options
    if isinstance(context.node_config, dict) and "partition-id" in context.node_config:
        partition_id = context.node_config["partition-id"]
    elif hasattr(context, "client_index"):
        # Use client_index as fallback in newer versions
        partition_id = context.client_index
    else:
        # Default to a random partition for testing
        partition_id = 0
        
    # Convert to int if it's a string
    if isinstance(partition_id, str):
        partition_id = int(partition_id)
    
    # Get num_partitions similarly
    if isinstance(context.node_config, dict) and "num-partitions" in context.node_config:
        num_partitions = context.node_config["num-partitions"]
    else:
        # Use the global constant
        num_partitions = NUM_PARTITIONS
    
    if isinstance(num_partitions, str):
        num_partitions = int(num_partitions)
    
    print(f"Using partition_id: {partition_id}, num_partitions: {num_partitions}")
    
    trainloader, valloader, _ = load_datasets(partition_id=partition_id, num_partitions=num_partitions)

    # Create a single Flower client
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
    print("Server evaluate: starting")
    net = get_model()
    trainloader, valloader, testloader = load_datasets(0, NUM_PARTITIONS)
    print("Server evaluate: datasets loaded")

    # Handle empty or invalid parameters
    if not parameters or len(parameters) == 0:
        print("Warning: Empty parameters for evaluation, using initial model")
    else:
        try:
            set_parameters(net, parameters)
        except Exception as e:
            print(f"Error setting parameters for evaluation: {e}")
            # Continue with initial model

    device = get_device()
    net.to(device)

    # Evaluate
    loss, accuracy, precision, recall, f1 = test(net, testloader)
    tracker.update(accuracy, loss, precision, recall, f1)

    print(f"Server-side evaluation metrics - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, "
          f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }

def fit_config(server_round: int):
    """
    Generate a dynamic training configuration dictionary for each federated learning round.

    This function adjusts client-side training parameters based on the current server round:
    - The number of local epochs increases by 1 every 10 rounds (starting with 1 epoch at round 0).
    - The learning rate is set to 0.001 initially, and reduces to 0.0005 after round 50.

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
        "learning_rate": 0.001 if server_round < 50 else 0.0005  # Learning rate for EfficientNet
    }
    return config

# Main execution section
if __name__ == "__main__":
    print("Using SVHN dataset")

    # Instantiate tracker
    tracker = MetricsTracker()
    print("MODELGETTING")
    # Create a model and get parameters
    model = get_model()
    model = model.cpu()
    params = get_parameters(model)
    print("STATEGY")
    # Create the strategy
    global_strategy = TrustFedAvg(
        warmup_rounds=15,
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=NUM_PARTITIONS,
        initial_parameters=ndarrays_to_parameters(params),
        evaluate_fn=evaluate,
        on_fit_config_fn=fit_config,
    )

    print("SIMULATION")
    # Use run_simulation from flwr.simulation
    start_simulation(
        client_fn=client_fn,
        num_clients=NUM_PARTITIONS,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=global_strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0}
    )
    print("METRICS")
    # Print metrics
    tracker.report()

    # Save everything
    metrics = {
        "accuracy": tracker.centralized_accuracy,
        "loss": tracker.centralized_loss,
        "precision": tracker.centralized_precision,
        "recall": tracker.centralized_recall,
        "f1": tracker.centralized_f1
    }

    os.makedirs("./saved_models", exist_ok=True)
    with open("./saved_models/fed_metrics_efficientnet.json", "w") as f:
        json.dump(metrics, f)

    if hasattr(global_strategy, "trust_log"):
        plt.plot(global_strategy.trust_log, label="Average Trust", linewidth=2, color="black")

    if hasattr(global_strategy, "client_trust_log"):
        for cid, history in global_strategy.client_trust_log.items():
            plt.plot(history, label=f"{cid}", linestyle="--", alpha=0.8)

        with open("./saved_models/trust_scores.json", "w") as f:
            json.dump({
                "avg_trust": global_strategy.trust_log,
                "client_trust": global_strategy.client_trust_log
            }, f)

    plt.xlabel("Round")
    plt.ylabel("Trust Score")
    plt.title("Trust Score Evolution Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("trust_combined_plot.png")

    torch.save(model.state_dict(), "./saved_models/fed_model_efficientnet_svhn.pth")