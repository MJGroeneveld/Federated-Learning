from collections import OrderedDict


import matplotlib.pyplot as plt 
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl 
import torchmetrics

import torch.utils.data as data

# from torchvision.datasets import CIFAR10
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datasets.utils.logging import disable_progress_bar

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import flwr as fl

from functions import create_model, Net, load_dataset

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Training on {DEVICE}")
print(f"Flower {fl.__version__} / PyTorch {torch.__version__}")
# disable_progress_bar()

class ClassifierCIFAR10(pl.LightningModule): 
    def __init__(self, config): 
        super().__init__() 

        #defining model 
        self.dataset_name   = config['dataset_name']
        self.num_classes    = config['num_classes']
        self.batch_size     = config['batch_size']
        self.ooddataset_name= config['ooddataset_name']
        self.net            = Net(num_classes=self.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

        # buffer om outputs op te slaan
        self.test_outputs = []

    def training_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.train_acc(outputs, labels)
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_acc", self.train_acc, prog_bar=False, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.val_acc(outputs, labels)
        self.log("val_loss", loss, prog_bar=False)
        self.log("val_acc", self.val_acc, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch["img"]
        labels = batch.get("label", None)

        if labels is not None: 
            labels = labels

        outputs = self(images)
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)

        # Loss alleen berekenen als labels beschikbaar zijn (ID)
        if labels is not None:
            loss = self.criterion(outputs, labels)
            self.test_acc(preds, labels)
            self.log("test_acc", self.test_acc, prog_bar=False, on_epoch=True, on_step=False)
            self.log("test_loss", loss, prog_bar=False)
        else:
            # Geen labels (OOD), geen loss/logging
            loss = torch.tensor(0.0, device=images.device)

        # if self.dataset_name == 'cifar10':
        #     # Alleen voor in-distribution evalueren we accuracy
        #     self.test_acc(preds, labels)
        #     self.log("test_acc", self.test_acc, prog_bar=False, on_epoch=True, on_step=False)
        
        # self.log("test_loss", loss, prog_bar=False)

        self.test_outputs.append({
            "loss": loss.detach(),
            "correct": (preds == labels).sum().item() if labels is not None else None,
            "total": labels.size(0) if labels is not None else None,
            "max_prob": probs.max(dim=1).values,
        })
        return loss
    
    def forward(self, X): 
        return self.net(X)

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
def main(config): 
    trainloader, valloader, _ = load_dataset(config)
    model = ClassifierCIFAR10(config) 
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    
    trainer = pl.Trainer(  
        max_epochs=config['max_epochs'], 
        accelerator="gpu", 
        callbacks=[checkpoint_callback],
        devices=1, 
        default_root_dir=config['bin'],)
    
    trainer.fit(model, trainloader, valloader)  # valloader optioneel
    
if __name__ == '__main__':
    config = { 
    'batch_size'        : 32, 
    'optimizer_lr'      : 0.0001, 
    'max_epochs'        : 50,  
    'bin'               : 'centralized_learning', 
    'experiment_name'   : '50_epochs_cifar10_cifar100',
    'num_clients'       : 1,
    'num_classes'       : 10,
    'partition_id'      : 0,
    'dataset_name'      : 'cifar10', 
    'ooddataset_name'   : 'cifar100'
    }
    main(config)