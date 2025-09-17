import torchvision 
import os
from collections import OrderedDict
import numpy as np
import torch.nn as nn 
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import torch 
import torch.nn.functional as F
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

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

    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})

    # if not ood:
    partition = fds.load_partition(partition_id, "train")

    # 20% for on federated evaluation (testset)
    partition_full = partition.train_test_split(test_size=0.2, seed=42)

    # 60 % for the federated train and 20 % for the federated validation (both in fit)
    partition_train_valid = partition_full["train"].train_test_split(
        train_size=0.75, seed=42
    )

    partition_train = partition_train_valid["train"].with_transform(apply_train_transforms)
    partition_val   = partition_train_valid["test"].with_transform(apply_test_transforms)
    
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = DataLoader(partition_val, batch_size=batch_size, num_workers=2)

    partition_test = partition_full["test"].with_transform(apply_test_transforms) 
    testloader = DataLoader(partition_test, batch_size=batch_size, num_workers=2)

    # print(f"📊 Dataset sizes (partition {partition_id}/{num_partitions}):")
    # print(f"  Train set: {len(partition_train)} samples") #30000 in centralized learning and 9999 in federated learning with 3 clients
    # print(f"  Val set:   {len(partition_val)} samples") #10000 in centralized learning and 3334 in federated learning
    # print(f"  Test set:  {len(partition_test)} samples") # 10000 in centralized learning and 3334 in federated learning

    # print(f"  Trainloader batches: {len(trainloader)} (batch_size={trainloader.batch_size})") # 938 in centralized learning and 313 in federated learning
    # print(f"  Valloader batches:   {len(valloader)} (batch_size={valloader.batch_size})") # 313 in centralized learning and 105 in federated learning
    # print(f"  Testloader batches:  {len(testloader)} (batch_size={testloader.batch_size})") # 313 in centralized learning and 105 in federated learning

    return trainloader, valloader, testloader    

def evaluate_ood(config, model, id_loader, ood_loader, trainer, out_dir: str, tag: str = ""):
    # ---- In-distribution evaluatie ----
    model.test_outputs.clear()
    trainer.test(model, id_loader)
    id_msps = torch.cat([x["max_prob"] for x in model.test_outputs])

    # id_outputs = []  # aparte lijst voor ID
    # model.test_outputs = id_outputs  # model vult deze tijdens test_step
    # trainer.test(model, id_loader, verbose=False)
    # id_msps = torch.cat([x["max_prob"] for x in id_outputs])

    # ---- OOD evaluatie ----
    model.test_outputs.clear()
    trainer.test(model, ood_loader)
    ood_msps = torch.cat([x["max_prob"] for x in model.test_outputs])
    
    # ood_outputs = []  # aparte lijst voor OOD
    # model.test_outputs = ood_outputs
    # trainer.test(model, ood_loader, verbose=False)
    # ood_msps = torch.cat([x["max_prob"] for x in ood_outputs])

    # ---- AUROC berekenen ----
    # labels: 1 = ID, 0 = OOD
    y_true = torch.cat([torch.ones_like(id_msps), torch.zeros_like(ood_msps)])
    y_score = torch.cat([id_msps, ood_msps])

    auroc = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
    print(f"AUROC (ID vs OOD via MSP): {auroc:.4f}")

    # ---- Histogram plotten ----
    os.makedirs(out_dir, exist_ok=True)
    filename = f"ood_hist_{tag}.png" if tag else "ood_hist.png"
    filepath = os.path.join(out_dir, filename)

    plt.figure()
    plt.hist(id_msps.cpu().numpy(), bins=50, alpha=0.6, label="ID (CIFAR-10)")
    plt.hist(ood_msps.cpu().numpy(), bins=50, alpha=0.6, label="OOD (CIFAR-100)")
    plt.xlabel("Maximum Softmax Probability (MSP)")
    plt.ylabel("Aantal samples")
    plt.legend()
    plt.title(f"MSP distributie: ID vs OOD (AUROC={auroc:.3f})")
    plt.savefig(filepath)
    plt.close()

    return id_msps, ood_msps, auroc