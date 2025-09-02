import torchvision 
import torch.nn as nn 
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset
from flwr_datasets import FederatedDataset
import torch 
import torch.nn.functional as F
from typing import List, Tuple

def create_model():
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class Net(nn.Module):
    def __init__(self, num_classes=10) -> None:
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
    
def load_dataset(partition_id: int = 0, num_clients: int = 10, batch_size: int = 32, dataset_name: str = "cifar10") -> Tuple[List[DataLoader], List[DataLoader], DataLoader]: 
    DATA_MEANS = (0.5, 0.5, 0.5)
    DATA_STD = (0.5, 0.5, 0.5) 
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
    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [test_transform(img) for img in batch["img"]]
        return batch
    
    fds = FederatedDataset(dataset=dataset_name, partitioners={"train": num_clients})
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloader, valloader, testloader    