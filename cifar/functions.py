from collections import OrderedDict
from typing import List
import numpy as np
import torch.nn as nn 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from torchvision.datasets import CIFAR100
from flwr_datasets.partitioner import IidPartitioner
import torch 
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1) # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
      self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1) # We double the feature maps for every conv layer as in pratice it is really good.
      self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
      self.fc1 = nn.Linear(4*4*64, 500) # I/p image size is 32*32, after 3 MaxPooling layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
      self.dropout1 = nn.Dropout(0.5)
      self.fc2 = nn.Linear(500, num_classes) # output nodes are 10 because our dataset have 10 different categories
    def forward(self, x):
      x = F.relu(self.conv1(x)) #Apply relu to each output of conv layer.
      x = F.max_pool2d(x, 2, 2) # Max pooling layer with kernal of 2 and stride of 2
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv3(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 4*4*64) # flatten our images to 1D to input it to the fully connected layers
      x = F.relu(self.fc1(x))
      x = self.dropout1(x) # Applying dropout b/t layers which exchange highest parameters. This is a good practice
      x = self.fc2(x)
      return x

class Net(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def apply_train_transforms(batch):
    # Instead of passing transforms to CIFAR10(..., transform=transform)
    # we will use this function to dataset.with_transform(apply_transforms)
    # The transforms object is exactly the same
    DATA_MEANS = (0.5, 0.5, 0.5)
    DATA_STD = (0.5, 0.5, 0.5) 
    train_transform = transforms.Compose([
                                    # transforms.CenterCrop(256), 
                                    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # do some small translations --> small (up, right, left, down)
                                    transforms.ColorJitter(brightness=0.2),  # change the brightness of the images 
                                    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Also do gaussian blur
                                    transforms.ToTensor(),
                                    transforms.Normalize(DATA_MEANS, DATA_STD)
                                    ])
    batch["img"] = [train_transform(img) for img in batch["img"]]
    return batch

def apply_test_transforms(batch):
    DATA_MEANS = (0.5, 0.5, 0.5)
    DATA_STD = (0.5, 0.5, 0.5) 
    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(DATA_MEANS, DATA_STD)
                                        ])
    batch["img"] = [test_transform(img) for img in batch["img"]]
    return batch

def get_parameters(model) -> List[np.ndarray]:
    # Return model parameters as a list of NumPy ndarrays
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters: List[np.ndarray]) -> None:
    # Set model parameters from a list of NumPy ndarrays
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

fds = None  # Cache FederatedDataset

def load_dataset(partition_id: int=0, num_partitions: int=1, batch_size: int=32): 
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:                                                                     #als fds nog niet bestaat, wordt er een nieuwe federated dataset aangemaakt
        partitioner = IidPartitioner(num_partitions=num_partitions)                     #de dataset wordt IID gesplitst over num_partitions clients 
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})

    partition = fds.load_partition(partition_id, "train")                               #haalt de data op voor een speicifieke client (parition_id) uit de train set

    # 20% for on federated evaluation (testset)
    partition_full = partition.train_test_split(test_size=0.2, seed=42)                 #partition_full["train"] → 80% van de data & partition_full["test"] → 20% van de data

    # van de 80% train dataset, 75% voor training en 25% voor validation
    partition_train_valid = partition_full["train"].train_test_split(
        train_size=0.75, seed=42
    )

    partition_train = partition_train_valid["train"].with_transform(apply_train_transforms)
    partition_val   = partition_train_valid["test"].with_transform(apply_test_transforms)
    
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(partition_val, batch_size=batch_size, num_workers=0)

    partition_test = partition_full["test"].with_transform(apply_test_transforms) 
    testloader = DataLoader(partition_test, batch_size=batch_size, num_workers=0)

    # print(f"📊 Dataset sizes (partition {partition_id}/{num_partitions}):")
    # print(f"  Train set: {len(partition_train)} samples") #30000 in centralized learning and 9999 in federated learning with 3 clients
    # print(f"  Val set:   {len(partition_val)} samples") #10000 in centralized learning and 3334 in federated learning
    # print(f"  Test set:  {len(partition_test)} samples") # 10000 in centralized learning and 3334 in federated learning

    # print(f"  Trainloader batches: {len(trainloader)} (batch_size={trainloader.batch_size})") # 938 in centralized learning and 313 in federated learning
    # print(f"  Valloader batches:   {len(valloader)} (batch_size={valloader.batch_size})") # 313 in centralized learning and 105 in federated learning
    # print(f"  Testloader batches:  {len(testloader)} (batch_size={testloader.batch_size})") # 313 in centralized learning and 105 in federated learning

    return trainloader, valloader, testloader    

def load_dataset_ood(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    testset_ood = CIFAR100(root="./data", train=False, download=True, transform=transform)
    testloader_ood = DataLoader(testset_ood, batch_size=batch_size, shuffle=False, num_workers=0)
    return testloader_ood