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
disable_progress_bar()

class ClassifierCIFAR10(pl.LightningModule): 
    def __init__(self, net): 
        super().__init__() 

        #defining model 
        self.net = net
        self.batch_size = 32
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def training_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.train_acc(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.val_acc(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.test_acc(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True, on_epoch=True, on_step=False)

        return loss
    
    def forward(self, X): 
        return self.net(X)

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    
def main(): 
    trainloader, valloader, testloader = load_dataset(partition_id=0)
    net = Net().to(DEVICE)
    model = ClassifierCIFAR10(net) 
    trainer = pl.Trainer(max_epochs=5, accelerator="gpu", devices=1)
    trainer.fit(model, trainloader, valloader)  # valloader optioneel
    trainer.test(model, testloader)
    
if __name__ == '__main__':
    main()