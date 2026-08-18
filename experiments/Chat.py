from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
import warnings
from torch.utils.data import DataLoader

import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Context, NDArrays, Scalar, Parameters
from flwr.simulation import run_simulation
from datasets import load_dataset
from flwr_datasets import FederatedDataset

warnings.filterwarnings("ignore", category=DeprecationWarning)

DEVICE = torch.device("cpu")
print(f"Training on {DEVICE}")

NUM_CLIENTS = 50
NUM_PARTITIONS = 50
NUM_ROUNDS = 100
BATCH_SIZE = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FlowerClient(NumPyClient):
    def __init__(self, pid, net, trainloader, valloader):
        self.pid = pid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, config["local_epochs"])
        self.net = structured_prune_model(self.net, amount=0.3)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

class AdaptiveFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        total_samples = sum(fit_res.num_examples for _, fit_res in results)
        weighted_updates = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        aggregated = [sum(layer * (num_samples / total_samples) for layer, num_samples in zip(layers, [w[1] for w in weighted_updates])) for layers in zip(*[w[0] for w in weighted_updates])]
        return ndarrays_to_parameters(aggregated), {}

def structured_prune_model(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    return model

def train(net, trainloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["fine_label"].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            print(batch.keys())
            images, labels = batch["img"].to(DEVICE), batch["fine_label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss / len(testloader.dataset), correct / total

def load_datasets(partition_id, num_partitions):
    fds = FederatedDataset(dataset="cifar100", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id).train_test_split(test_size=0.2, seed=42)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainloader = DataLoader(partition["train"].with_transform(lambda b: {"img": [transform(img) for img in b["img"]]}), batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition["test"].with_transform(lambda b: {"img": [transform(img) for img in b["img"]]}), batch_size=BATCH_SIZE)
    return trainloader, valloader

def set_parameters(net, parameters):
    state_dict = net.state_dict()
    new_state_dict = OrderedDict((k, torch.Tensor(v)) for k, v in zip(state_dict.keys(), parameters))
    net.load_state_dict(new_state_dict, strict=False)

def get_parameters(net):
    return [v.cpu().numpy() for _, v in net.state_dict().items()]

def server_fn(context):
    return ServerAppComponents(strategy=AdaptiveFedAvg(fraction_fit=0.3, fraction_evaluate=0.3, min_fit_clients=3, min_evaluate_clients=3, min_available_clients=NUM_PARTITIONS, initial_parameters=ndarrays_to_parameters(get_parameters(Net())), evaluate_fn=evaluate, on_fit_config_fn=fit_config), config=ServerConfig(num_rounds=NUM_ROUNDS))

def evaluate(server_round, parameters, config):
    net = Net().to(DEVICE)
    _, valloader = load_datasets(0, NUM_PARTITIONS)
    set_parameters(net, parameters)
    return test(net, valloader)

def fit_config(server_round):
    return {"server_round": server_round, "local_epochs": 1 + server_round // 10, "learning_rate": 0.01 if server_round < 50 else 0.005}

load_dataset("cifar100")
client = ClientApp(client_fn=lambda ctx: FlowerClient(ctx.node_config["partition-id"], Net().to(DEVICE), *load_datasets(ctx.node_config["partition-id"], NUM_PARTITIONS)).to_client())
server = ServerApp(server_fn=server_fn)
run_simulation(server, client, num_supernodes=NUM_PARTITIONS)
