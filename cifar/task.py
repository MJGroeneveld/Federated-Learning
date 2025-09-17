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

# logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
# print(f"Training on {DEVICE}")
# print(f"Flower {fl.__version__} / PyTorch {torch.__version__}")
# disable_progress_bar()

class ClassifierCIFAR10(pl.LightningModule): 
    def __init__(self): 
        super().__init__() 

        # self.num_classes    = config['num_classes']
        self.model            = Net() #config['model'](num_classes = self.num_classes) 

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
        self.log("train_loss", loss)#, prog_bar=False, on_epoch=True, on_step=False)
        self.log("train_acc", accurancy)# , prog_bar=False, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch["img"], batch["label"]
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        accuracy = self.val_acc(outputs, labels)
        self.log("val_loss", loss)#, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_acc", accuracy)#, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        images = batch["img"]
        labels = batch.get("label", None)

        if labels is not None: 
            labels = labels

        outputs = self.model(images)
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)

        # Loss alleen berekenen als labels beschikbaar zijn (ID)
        if labels is not None:
            loss = self.criterion(outputs, labels)
            accuracy = self.test_acc(preds, labels)
            self.log("test_acc", accuracy)#, prog_bar=True, on_epoch=True, on_step=False)
            self.log("test_loss", loss)#, prog_bar=True, on_epoch=True, on_step=False)
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