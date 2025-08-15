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

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

import flwr as fl
from flwr.common import Metrics

if torch.backends.mps.is_available():
    device = torch.device('mps')  # Apple Metal Performance Shaders
elif torch.cuda.is_available() :
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

num_clients = 10 
batch_size  = 32 


class CIFAR10DataModule(pl.LightningDataModule): 
    def __init__ (self, config): 
        super().__init__() 
        self.train_dir    = config['train_dir']
        self.train_labels = config['train_labels']
        self.batch_size   = config['batch_size']
        self.num_clients  = config['num_clients']
        
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
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, num_workers=2)

