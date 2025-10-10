import matplotlib.pyplot as plt 

import os 

import torch 

from flwr.common import Context
import pytorch_lightning as pl 

from torchvision import datasets

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import flwr as fl
from flwr.client import Client, ClientApp

from datasets.utils.logging import disable_progress_bar

from cifar.functions import load_dataset, load_dataset_ood, get_parameters, set_parameters
from cifar.task import ClassifierCIFAR10

from sklearn.metrics import roc_auc_score

disable_progress_bar()
# logging.basicConfig(level=logging.INFO)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, valloader, testloader, oodtestloader=None):
        self.model          = ClassifierCIFAR10()
        self.cid            = cid
        self.trainloader    = trainloader
        self.valloader      = valloader
        self.testloader     = testloader
        self.oodtestloader  = oodtestloader

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        set_parameters(self.model, parameters)
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
        logger = CSVLogger(save_dir=config['bin'], name=f"client{self.cid}_{config['experiment_name']}", version=f"round{config['current_round']}")

        trainer = pl.Trainer(max_epochs=config['local_epochs'], 
                            default_root_dir = config['bin'], 
                            callbacks=[checkpoint_callback],
                            logger=logger,
                            enable_progress_bar=False) 
        
        trainer.fit(self.model, self.trainloader, self.valloader)
        
        #len(self.trainloader.dataset) = 9999
        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_parameters(self.model, parameters)
        trainer = pl.Trainer(
                        default_root_dir=config['bin'],
                        logger=False,
                        enable_progress_bar=False)

       
        # Default return values
        metrics = {}

        # ID dataset
        results_id = trainer.test(self.model, self.testloader)
        loss = results_id[0]['test_loss']
        accuracy = results_id[0]['test_acc']
        metrics["accuracy"] = float(accuracy)

        # OOD dataset
        if self.oodtestloader: 
            trainer.test(self.model, self.oodtestloader)
            all_probs_id = torch.cat([x["max_prob"] for x in self.model.test_outputs if "max_prob" in x]).cpu() #length = 3334
            all_probs_ood = torch.cat([x["max_prob_ood"] for x in self.model.test_outputs if "max_prob_ood" in x]).cpu() #length is 10000
            # Labels: 0 = ID, 1 = OOD
            y_true = torch.cat([
                torch.zeros_like(all_probs_id, dtype=torch.int32),
                torch.ones_like(all_probs_ood, dtype=torch.int32),
            ])
            y_score = torch.cat([all_probs_id, all_probs_ood])

            auroc = roc_auc_score(y_true.numpy(), (y_score.numpy()))
            metrics["auroc_ood"] = float(auroc)

            # ---- Histogram plotten ----
            out_dir = os.path.join(
                config['bin'],
                f"client{self.cid}_{config['experiment_name']}",
                f"round{config['current_round']}"
            )
            os.makedirs(out_dir, exist_ok=True)
            print(f"Saving plots for client {self.cid} in {out_dir}")

            filename = f"ood_hist_client{self.cid}.png"
            filepath = os.path.join(out_dir, filename)

            plt.figure()
            plt.hist(all_probs_id.numpy(), bins=50, alpha=0.6, label="ID (CIFAR-10)")
            plt.hist(all_probs_ood.numpy(), bins=50, alpha=0.6, label="OOD (CIFAR-100)")
            plt.xlabel("Maximum Softmax Probability (MSP)")
            plt.ylabel("Aantal samples")
            plt.legend()
            plt.title(f"MSP distributie: ID vs OOD (AUROC={auroc:.3f})")
            plt.savefig(filepath)
            plt.close()

        return float(loss), len(self.testloader.dataset), metrics

def client_fn(context: Context) -> Client: 
    """Create a Flower client representing a single organization."""
    # Read the node_config to fetch data partition associated to this node
    partition_id                = context.node_config["partition-id"]
    num_partitions              = context.node_config["num-partitions"]
    batch_size                  = 32
    trainloader, valloader, testloader  = load_dataset(partition_id=partition_id, num_partitions=num_partitions, batch_size=batch_size)
    testloader_ood              = load_dataset_ood(batch_size=batch_size)

    # Read run_config to fetch hyperparameters relevant to this run
    return FlowerClient(partition_id, trainloader, valloader, testloader, testloader_ood).to_client()

# Create the ClientApp
client = ClientApp(client_fn=client_fn)
