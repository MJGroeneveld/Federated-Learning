from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt 
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from torchmetrics.functional import accuracy

import flwr as fl
import cifar
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.common import Context, Metrics

from functions import create_model, Net, load_dataset
from cifar import ClassifierCIFAR10

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, testloader): 
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1, enable_checkpointing=False,logger=False,enable_progress_bar=False) 
        model = ClassifierCIFAR10(self.net) 
        trainer.fit(model, self.trainloader, self.valloader)
        return self.get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1, enable_checkpointing=False,logger=False,enable_progress_bar=False)
        model = ClassifierCIFAR10(self.net) 
        result = trainer.test(model, self.valloader, verbose=False)
        loss = result[0]['test_loss']  
        accuracy = result[0]['test_acc']
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
  
def client_fn(context: Context) -> Client: 
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, testloader = load_dataset(partition_id=partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader, testloader).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=10,
        min_evaluate_clients=5,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    )

    # Configure the server for 5 rounds of training
    config = ServerConfig(num_rounds=5)

    return ServerAppComponents(strategy=strategy, config=config)

if __name__ == "__main__":
    # Create a new server instance with the updated FedAvg strategy
    server = ServerApp(server_fn=server_fn)
    NUM_CLIENTS = 10 

    # Specify the resources each of your clients need
    # By default, each client will be allocated 1x CPU and 0x GPUs
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

    # When running on GPU, assign an entire GPU for each client
    if DEVICE == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`
    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_CLIENTS,
        backend_config=backend_config,
    )