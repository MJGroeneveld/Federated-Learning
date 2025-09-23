from collections import OrderedDict

import torch 
import logging
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
import torchmetrics

from torch.optim.adam import Adam

from typing import Any

import torch.utils.data as data
from pytorch_lightning.loggers import CSVLogger

# from torchvision.datasets import CIFAR10
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datasets.utils.logging import disable_progress_bar

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import flwr as fl

from cifar.functions import Net, load_dataset, LeNet

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

class ClassifierCIFAR10(pl.LightningModule): 
    def __init__(self): 
        super().__init__() 

        self.model      = Net()
        self.criterion  = torch.nn.CrossEntropyLoss()
        self.train_acc  = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc    = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc   = torchmetrics.Accuracy(task="multiclass", num_classes=10)

        # buffer om outputs op te slaan
        self.test_outputs = []

    def training_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["label"]
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        accurancy = self.train_acc(outputs, labels)
        self.log("train_loss", loss)
        self.log("train_acc", accurancy)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["label"]
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        accuracy = self.val_acc(outputs, labels)
        self.log("val_loss", loss)
        self.log("val_acc", accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        if isinstance(batch, dict): # FederatedDataset CIFAR10 ID dataset
            images, labels = batch["img"], batch.get("label", None)
            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)

            loss = self.criterion(outputs, labels)
            accuracy = self.test_acc(preds, labels)
            self.log("test_acc", accuracy)
            self.log("test_loss", loss)

            self.test_outputs.append({
                # "loss": loss.detach(),
                # "correct": (preds == labels).sum().item(), 
                # "total": labels.size(0), 
                "max_prob": probs.max(dim=1).values,
            })

        if isinstance(batch, list):  # torchvision CIFAR100 OOD dataset
            images_ood, labels_ood = batch
            outputs_ood = self.model(images_ood)
            probs_ood = torch.softmax(outputs_ood, dim=1)

            loss = torch.tensor(0.0, device=images_ood.device)

            self.test_outputs.append({
                "labels_ood": labels_ood, 
                "max_prob_ood": probs_ood.max(dim=1).values,
            })
        return loss
    
    def forward(self, X): 
        return self.model(X)

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
# def main(config): 
#     trainloader, valloader, _ = load_dataset()
#     model = ClassifierCIFAR10() 
#     checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
#     logger = CSVLogger(save_dir=config['bin'], version=config['experiment_name'])
#     trainer = pl.Trainer(  
#         max_epochs=config['max_epochs'], 
#         accelerator="gpu", 
#         callbacks=[checkpoint_callback],
#         devices=1, 
#         logger=logger,
#         default_root_dir=config['bin'],)
    
#     trainer.fit(model, trainloader, valloader)  # valloader optioneel
    
# if __name__ == '__main__':
#     config = { 
#     'model'             : LeNet, #Net,
#     'batch_size'        : 32, 
#     'max_epochs'        : 2,  
#     'bin'               : 'centralized_learning', 
#     'experiment_name'   : '100_epochs_LeNet_cifar10_cifar100',
#     'num_clients'       : 1,
#     'num_classes'       : 10,
#     'dataset_name'      : 'cifar10', 
#     'ooddateset_name'   : 'cifar100'
#     }
#     main(config)