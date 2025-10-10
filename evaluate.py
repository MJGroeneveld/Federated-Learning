import os
import torch
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from cifar.task import ClassifierCIFAR10
from cifar.functions import load_dataset, load_dataset_ood  # jouw functies

def analyze_checkpoint(ckpt_path, client_id, round_id, out_dir):
    # Model laden
    model = ClassifierCIFAR10.load_from_checkpoint(ckpt_path)
    model.eval()

    trainer = pl.Trainer(logger=False, enable_progress_bar=False)

    # Data opnieuw laden
    _, _, id_loader = load_dataset(partition_id=int(client_id.replace("client","")), num_partitions=3, batch_size=32)
    ood_loader = load_dataset_ood(batch_size=32)

    # Test op ID
    trainer.test(model, id_loader)
    all_probs_id = torch.cat([x["max_prob"] for x in model.test_outputs if "max_prob" in x]).cpu()

    # Test op OOD
    trainer.test(model, ood_loader)
    all_probs_ood = torch.cat([x["max_prob_ood"] for x in model.test_outputs if "max_prob_ood" in x]).cpu()

    # ---- AUROC berekenen ----
    y_true = torch.cat([
        torch.zeros_like(all_probs_id, dtype=torch.int32),
        torch.ones_like(all_probs_ood, dtype=torch.int32),
    ])
    y_score = torch.cat([all_probs_id, all_probs_ood])
    auroc = roc_auc_score(y_true.numpy(), y_score.numpy())
    print(f"Client {client_id}, Round {round_id}, AUROC={auroc:.3f}")

    # ---- Histogram ----
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.hist(all_probs_id.numpy(), bins=50, alpha=0.6, label="ID")
    plt.hist(all_probs_ood.numpy(), bins=50, alpha=0.6, label="OOD")
    plt.xlabel("MSP")
    plt.ylabel("Aantal samples")
    plt.legend()
    plt.title(f"MSP distributie (AUROC={auroc:.3f})")
    plt.savefig(os.path.join(out_dir, f"ood_hist_{client_id}_round{round_id}.png"))
    plt.close()

    # ---- Confusion Matrix ----
    preds = (y_score < 0.5).int()  # threshold 0.5
    cm = confusion_matrix(y_true.numpy(), preds.numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ID", "OOD"])
    disp.plot()
    plt.title(f"Confusion Matrix ({client_id}, {round_id})")
    plt.savefig(os.path.join(out_dir, f"confmat_{client_id}_round{round_id}.png"))
    plt.close()

    return auroc

if __name__ == "__main__":
    base_dir = "ood_federated_learning"   # jouw bin folder
    experiment = "23092025"

    results = [] 

    for client in os.listdir(base_dir):
        if not client.endswith(experiment):  # skip andere dingen
            continue
        client_path = os.path.join(base_dir, client)

        for round_dir in os.listdir(client_path):
            round_path = os.path.join(client_path, round_dir, "checkpoints")
            if not os.path.exists(round_path):
                continue

            ckpts = [f for f in os.listdir(round_path) if f.endswith(".ckpt")]
            if not ckpts:
                continue

            # pak beste checkpoint (eerste in lijst of bv. laatste)
            ckpt_path = os.path.join(round_path, ckpts[0])
            auroc = analyze_checkpoint(
                ckpt_path,
                client_id=client.split("_")[0],   # bv. "client0"
                round_id=round_dir,               # bv. "round1"
                out_dir=client_path
            )

            results.append({
                "client": client,
                "round": round_dir,
                "auroc": auroc
            })

    if results:
        df = pd.DataFrame(results)
        avg_auroc = df["auroc"].mean()
        print(f"\n Gemiddelde AUROC over alle clients/rondes: {avg_auroc:.4f}")
