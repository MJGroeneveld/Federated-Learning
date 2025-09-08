from cifar import ClassifierCIFAR10
from functions import load_dataset, Net, evaluate_ood
import torch, os, glob
import pytorch_lightning as pl 
from oodeel.methods import MLS
from oodeel.datasets import load_data_handler
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cleanlab.outlier import OutOfDistribution


if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

def main(config): 
    # Load datasets 
    _,_, idtestloader  = load_dataset(config, ood=False)
    _,_, oodtestloader = load_dataset(config, dataset_name=config['ooddataset_name'], ood=True)

    # Load model from checkpoint 
    checkpoint_folder_path = f"/Users/melaniegroeneveld/Documents/Flower Federated Learning/Federated-Learning/centralized_learning/lightning_logs/version_0/checkpoints/"
    PATH = glob.glob(os.path.join(checkpoint_folder_path, '*.ckpt'))
    model = ClassifierCIFAR10.load_from_checkpoint(PATH[0], map_location=DEVICE, config=config)
   
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'], 
        accelerator="gpu", 
        devices=1, 
        default_root_dir=config['bin'],
        inference_mode=False)

    #Evaluate on in-distribution test set (CIFAR-10)
    evaluate_ood(model, idtestloader, oodtestloader, trainer)


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