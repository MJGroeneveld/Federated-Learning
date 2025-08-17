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
from flwr.common import Context

from functions import create_model

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainer, train_loader, val_loader, test_loader): 
        self.model = net 
        self.trainloader = train_loader
        self.valloader= val_loader
        self.testloader = test_loader
        self.trainer = trainer

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config): 
        # Set model parameters, train model and return updated model parameters 
        self.set_parameters(parameters)
        self.trainer.fit(model = self.model, train_dataloaders=self.trainloader, val_dataloaders=self.valloader)
        return self.get_parameters(), len(self.trainloader), {} 
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        results = self.trainer.test(model = self.model, dataloaders=self.testloader)
        loss = results[0]["test_loss"]
        accuracy = results[0]["test_acc"]   
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}
  
def client_fn(cid:str) -> FlowerClient: 
    """Create a Flower client representing a single organization"""
    # cid = client ID 

    # Load datamodule
    trainloaders, valloaders, testloader = cifar.load_dataset()

    # Each client gets a different trainloader/valloader, so each client will train and evaluate on their own unique data 

    # Load model 
    model = cifar.ClassifierCIFAR10() 

    # Create Trainer (per client)
    trainer = pl.Trainer(
        max_epochs=5,
        logger=None,
        accelerator="mps" if torch.backends.mps.is_available() else "cpu",
        deterministic=True,
        log_every_n_steps=1
    )
    trainloader = trainloaders[int(cid)] 
    valloader = valloaders[int(cid)]

    return FlowerClient(net = model, trainer = trainer, train_loader = trainloader, val_loader = valloader, test_loader = testloader)
    # Create a single Flower client represeting a single organization
    #client = FlowerClient(net = model, trainer = trainer, train_loader = trainloader, val_loader = valloader, test_loader = testloader)
    #fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

if __name__ == "__main__":
    # Define the strategy for federated learning
    # This strategy will be used by the server to aggregate updates from clients
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,  # Fraction of clients to use for training
        fraction_evaluate=0.5,  # Fraction of clients to use for evaluation
        min_fit_clients=2,  # Minimum number of clients to use for training
        min_evaluate_clients=2,  # Minimum number of clients to use for evaluation
        min_available_clients=2,  # Minimum number of clients to be available
    ) 

    fl.simulation.start_simulation(
        client_fn = client_fn,
        num_clients = 2,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy= strategy
    )