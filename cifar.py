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
from torchvision.datasets import CIFAR10
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import flwr as fl
from flwr.common import Context

from functions import create_model

def load_dataset(): 
    data_dir = 'data/cifar10'
    DATA_MEANS = (0.5, 0.5, 0.5)
    DATA_STD = (0.5, 0.5, 0.5) 
    num_clients = 2 #default is 1, but can be changed to more than 1 if there are multiple clients and we want to do federated learning 
    batch_size = 32
    test_transform = transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(DATA_MEANS, DATA_STD)
                                                  ])
    train_transform = transforms.Compose([#transforms.Resize((512, 512)), 
                                            transforms.CenterCrop(256), 
                                            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # do some small translations --> small (up, right, left, down)
                                            transforms.ColorJitter(brightness=0.2),  # change the brightness of the images 
                                            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Also do gaussian blur
                                            transforms.ToTensor(),
                                            transforms.Normalize(DATA_MEANS, DATA_STD)
                                            ])
    trainset = CIFAR10(data_dir, train=True, download=True, transform=train_transform)

    test_dataset  = CIFAR10(data_dir, train=False, download = True, transform=test_transform)

    partition_size = len(trainset) // num_clients #= 50000 / 25000
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, generator=torch.Generator().manual_seed(42))

    trainloaders = [] 
    valloaders = []
    for ds in datasets: 
        len_val = len(ds) // 10 
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        dataset_train, dataset_val = random_split(ds, lengths, generator=torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(dataset_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(dataset_val, batch_size=batch_size))
    testloader = DataLoader(test_dataset, batch_size = batch_size)#, num_workers=1)
    return trainloaders, valloaders, testloader

class ClassifierCIFAR10(pl.LightningModule): 
    def __init__(self): 
        super().__init__() 

        #defining model 
        self.model = create_model()

        self.batch_size     = 32

        self.test_step_y_prob = []
        self.test_step_y_prob_multiclass = []
        self.test_step_ys = []

        self.val_step_y_prob = [] 
        self.val_step_y_prob_multiclass = []
        self.val_step_ys = []

        self.batch_data_list = []

    def training_step(self, batch, batch_idx):
        #print("\nBATCH CONTENT:", type(batch)) #<class 'list'>
        X, y            = batch
        #X, y            = X.float().to(device), y.to(device)
        y_hat           = self.model(X)

        train_loss      = F.cross_entropy(y_hat, y)
        self.log(f'train_loss', train_loss, on_step=False, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        return {"loss": train_loss}
    
    def evaluate(self, batch, stage=None): 
        X, y            = batch
        y_hat           = self.model(X)

        loss            = F.cross_entropy(y_hat, y)

        y_pred          = torch.argmax(y_hat, dim=1) #find label with highest probability
        acc             = accuracy(y_pred, y, task='multiclass', num_classes=10)

        if stage: 
            self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
            self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        return {f"{stage}_loss": loss, "{stage}_acc": acc}
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage='val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')
    
    def forward(self, X): 
        out = self.model(X)
        return F.log_softmax(out, dim=1)

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer
    
def main(): 
    if torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Metal Performance Shaders
        accelerator = 'mps'    
    elif torch.cuda.is_available() :
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    trainloaders, valloaders, testloader = load_dataset()
    classifier          = ClassifierCIFAR10()    
    trainer             = pl.Trainer(max_epochs=5,
                                    logger=None, 
                                    accelerator=accelerator, 
                                    deterministic=True,
                                    log_every_n_steps=1)

    trainer.fit(classifier, train_dataloaders = trainloaders[0], val_dataloaders = valloaders[0])
    trainer.test(classifier, datamodule=testloader)
    
if __name__ == '__main__':
    main()