import torchvision 
import torch.nn as nn 
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from flwr_datasets import FederatedDataset
import torch 
import torch.nn.functional as F
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def create_model():
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

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

def load_dataset(config, dataset_name = None, ood=False):
    DATA_MEANS = (0.5, 0.5, 0.5)
    DATA_STD = (0.5, 0.5, 0.5) 
    test_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(DATA_MEANS, DATA_STD)
                                        ])
    train_transform = transforms.Compose([
                                        # transforms.CenterCrop(256), 
                                        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # do some small translations --> small (up, right, left, down)
                                        transforms.ColorJitter(brightness=0.2),  # change the brightness of the images 
                                        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Also do gaussian blur
                                        transforms.ToTensor(),
                                        transforms.Normalize(DATA_MEANS, DATA_STD)
                                        ])
    def apply_train_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [train_transform(img) for img in batch["img"]]
        return batch
    
    def apply_test_transforms(batch):
        batch["img"] = [test_transform(img) for img in batch["img"]]
        return batch
    
    dataset_name = dataset_name or config['dataset_name']
    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": config['num_clients']})

    if not ood:
        partition = fds.load_partition(config['partition_id'])
        partition_train_val = partition.train_test_split(test_size=0.2, seed=42)
        
        partition_train_val["train"] = partition_train_val["train"].with_transform(apply_train_transforms)
        partition_train_val["test"] = partition_train_val["test"].with_transform(apply_test_transforms)
        
        trainloader = DataLoader(partition_train_val["train"], batch_size=config['batch_size'], shuffle=True)
        valloader = DataLoader(partition_train_val["test"], batch_size=config['batch_size'])
    else: 
        trainloader, valloader = None, None

    testset = fds.load_split("test").with_transform(apply_test_transforms)
    testloader = DataLoader(testset, batch_size=config['batch_size'])

    return trainloader, valloader, testloader    

def evaluate_ood(model, id_loader, ood_loader, trainer):
    # ---- In-distribution evaluatie ----
    model.test_outputs.clear()
    id_results = trainer.test(model, id_loader)
    id_msps = torch.cat([x["max_prob"] for x in model.test_outputs])

    # ---- OOD evaluatie ----
    model.test_outputs.clear()
    ood_results = trainer.test(model, ood_loader)
    ood_msps = torch.cat([x["max_prob"] for x in model.test_outputs])

    # ---- Histogram plotten ----
    plt.hist(id_msps.cpu().numpy(), bins=50, alpha=0.6, label="ID (CIFAR-10)")
    plt.hist(ood_msps.cpu().numpy(), bins=50, alpha=0.6, label="OOD (CIFAR-100)")
    plt.xlabel("Maximum Softmax Probability (MSP)")
    plt.ylabel("Aantal samples")
    plt.legend()
    plt.title("MSP distributie: ID vs OOD")
    plt.show()

    # ---- AUROC berekenen ----
    # labels: 1 = ID, 0 = OOD
    y_true = torch.cat([torch.ones_like(id_msps), torch.zeros_like(ood_msps)])
    y_score = torch.cat([id_msps, ood_msps])

    auroc = roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy())
    print(f"AUROC (ID vs OOD via MSP): {auroc:.4f}")

    return id_msps, ood_msps, auroc