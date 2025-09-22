import matplotlib.pyplot as plt 

import os 
import warnings 
import torch 
import logging
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import Context
import pytorch_lightning as pl 
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets
from torchmetrics.functional import accuracy

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import flwr as fl
from flwr.client import Client, ClientApp, NumPyClient

from datasets.utils.logging import disable_progress_bar

from cifar.functions import  Net, load_dataset, LeNet, evaluate_ood, get_parameters, set_parameters
from cifar.task import ClassifierCIFAR10

disable_progress_bar()
# logging.basicConfig(level=logging.INFO)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, valloader, testloader):
        self.model          = ClassifierCIFAR10()
        self.cid            = cid
        self.trainloader    = trainloader
        self.valloader      = valloader
        self.testloader     = testloader

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        set_parameters(self.model, parameters)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
        logger = CSVLogger(save_dir=config['bin'], name=f"client{self.cid}_{config['experiment_name']}", version=f"round{config['current_round']}")

        trainer = pl.Trainer(max_epochs=config['local_epochs'], 
                            default_root_dir = config['bin'], 
                            callbacks=[checkpoint_callback],
                            logger=logger,
                            enable_progress_bar=False) 
        
        trainer.fit(self.model, self.trainloader, self.valloader)
        
        #len(self.trainloader.dataset) = 9999
        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_parameters(self.model, parameters)
        trainer = pl.Trainer(
                            default_root_dir=config['bin'],
                            logger=False,
                            enable_progress_bar=False)
        
        results = trainer.test(self.model, self.testloader)

        loss = results[0]['test_loss']  
        accuracy = results[0]['test_acc']

        # tag = f"client{self.partition_id}_round{self.config.get('round', 0)}"
        # out_dir = os.path.join(self.config['bin'], self.config['experiment_name'], "results")

       # _,_, auroc = evaluate_ood(self.config, self.net, self.testloader, self.oodtestloader, trainer, out_dir=out_dir, tag=tag)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}#, "auroc_ood": float(auroc)}

def client_fn(context: Context) -> Client: 
    """Create a Flower client representing a single organization."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = 32
    trainloader, valloader, testloader = load_dataset(partition_id=partition_id, num_partitions=num_partitions, batch_size=batch_size)

    # Read run_config to fetch hyperparameters relevant to this run
    return FlowerClient(partition_id, trainloader, valloader, testloader).to_client()

# Create the ClientApp
client = ClientApp(client_fn=client_fn)
