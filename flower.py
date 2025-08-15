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

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import flwr as fl
from flwr.common import Metrics

from functions import create_model

if torch.backends.mps.is_available():
    device = torch.device('mps')  # Apple Metal Performance Shaders
elif torch.cuda.is_available() :
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

class CIFAR10DataModule(pl.LightningDataModule): 
    def __init__ (self, data_dir: str = 'data/cifar10'): 
        super().__init__() 
        self.train_dir    = data_dir
        self.batch_size   = 64
        # self.num_clients  = config['num_clients']
        
        DATA_MEANS = (0.5, 0.5, 0.5)
        DATA_STD = (0.5, 0.5, 0.5) 
        self.test_transform = transforms.Compose([
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(DATA_MEANS, DATA_STD)
                                                  ])
        self.train_transform = transforms.Compose([#transforms.Resize((512, 512)), 
                                            transforms.CenterCrop(256), 
                                            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # do some small translations --> small (up, right, left, down)
                                            transforms.ColorJitter(brightness=0.2),  # change the brightness of the images 
                                            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Also do gaussian blur
                                            transforms.ToTensor(),
                                            transforms.Normalize(DATA_MEANS, DATA_STD)
                                            ])
        
    def prepare_data(self):
        # Download the dataset if it doesn't exist
        datasets.CIFAR10(self.train_dir, train=True, download=True)
        datasets.CIFAR10(self.train_dir, train=False, download=True)

    def setup(self, stage=None): 
        # Loading the CIFAR10 dataset. We need to split it into a training, test and validation part
        # We need to do a little trick because the valid and test set should not use the augmentation.    
        self.train_set = datasets.CIFAR10(self.train_dir, train=True, transform=self.train_transform)
        self.val_set   = datasets.CIFAR10(self.train_dir, train=True, transform=self.test_transform)
        self.test_set  = datasets.CIFAR10(self.train_dir, train=True, transform=self.test_transform)
        
        # We use the StratifiedShuffleSplit to split the data into train and validation sets
        sss = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state=42)
        X   = range(len(self.train_set)) #range(0, 6141)
        y   = [self.train_set[i][1] for i in X] #dan alles wat bij die 6141 hoort 

        self.client_trainloaders = []
        self.client_valloaders   = []

        for train_val_indices, test_indices in sss.split(X, y):
            # Split the data into train and validation sets
            train_val_dataset       = Subset(self.val_set, train_val_indices)
            train_val_dataset_aug   = Subset(self.train_set, train_val_indices)
            self.test_dataset       = Subset(self.test_set, test_indices)
            
            Xi      = range(len(train_val_dataset))
            labels  = [train_val_dataset[i][1] for i in Xi]

            # Splits train+val in train and val
            train_indices, val_indices = train_test_split(
                range(len(train_val_dataset)), train_size=0.88, stratify=labels, random_state=42)

            self.train_dataset  = Subset(train_val_dataset_aug, train_indices)
            self.val_dataset    = Subset(train_val_dataset, val_indices)
        
    def train_dataloader(self): 
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=1)

    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=1, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=1)

class Client(pl.LightningModule): 
    def __init__(self): 
        super().__init__() 

        #defining model 
        self.model = create_model()

        # self.experiment_name = config['experiment_name']

        #assigning optimizer values 
        # self.optimizer_name = config['optimizer_name']
        # self.lr             = config['optimizer_lr']

        self.batch_size     = 64

        self.test_step_y_prob = []
        self.test_step_y_prob_multiclass = []
        self.test_step_ys = []

        self.val_step_y_prob = [] 
        self.val_step_y_prob_multiclass = []
        self.val_step_ys = []

        self.batch_data_list = []
    
    def training_step(self, batch, batch_idx):
        X, y            = batch
        #X, y            = X.float().to(device), y.to(device)
        y_hat           = self.model(X)

        train_loss      = F.cross_entropy(y_hat, y)
        self.log(f'train_loss', train_loss, on_step=False, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        return {"loss": train_loss}
    
    def evaluate(self, batch, stage=None): 
        X, y            = batch
        y_hat           = self(X)

        loss            = F.cross_entropy(y_hat, y)

        y_pred          = torch.argmax(y_hat, dim=1) #find label with highest probability
        acc             = accuracy(y_pred, y, task='multiclass', num_classes=10)

        if stage: 
            self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
            self.log(f'{stage}_acc', acc, on_step=False, on_epoch=True, batch_size=self.batch_size, prog_bar=True)


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
    
def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader): 
        self.net = net 
        self.trainloader = trainloader 
        self.valloader = valloader 

    def get_parameters(self, config): 
        return get_parameters(self.net)
    
    def fit(self, parameters, config): 
        set_parameters(self.net, parameters)
        Client.training_step(self.net, self.trainloader) 
        return get_parameters(self.net), len(self.trainloader), {}
    
    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = Client.evaluate(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

if __name__ == '__main__':
    num_clients = 10 
    batch_size  = 64 
    pl.seed_everything(42)

    dm               = CIFAR10DataModule()
    model            = Client()
    trainer          = pl.Trainer(max_epochs=2,
                                    devices=1)

    trainer.fit(model, dm)
