import os
import torch
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from cifar.task import ClassifierCIFAR10
from cifar.functions import load_dataset, load_dataset_ood 

def compute_confident_thresholds(model, train_loader, num_classes):
    """
    Compute confident threshold vector c, where
    c_k = (1 / N_k) * sum_i p_{i,k} * 1[y_i = k]
    """
    model.eval()
    device = next(model.parameters()).device
    c_sums = torch.zeros(num_classes, device=device)
    c_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for batch in train_loader:
            x, y = batch["img"], batch["label"]
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)  # shape [B, K]

            for k in range(num_classes):
                mask = (y == k)
                if mask.any():
                    c_sums[k] += probs[mask, k].sum()
                    c_counts[k] += mask.sum()

    c = c_sums / c_counts.clamp(min=1)
    return c

def adjust_probs(probs, c):
    """ 
    Apply Cleanlab adjustment to predicted probabilities.
    p_tilde = (p - c + c_bar) / Z 
    """
    c = c.to(probs.device)
    c_bar = c.max()
    adjusted = probs - c.unsqueeze(0) + c_bar
    adjusted = torch.clamp(adjusted, min=0.0)  # veiligheidscheck
    Z = adjusted.sum(dim=1, keepdim=True)
    adjusted = adjusted / Z
    return adjusted

def compute_ood_scores(probs, c=None):
    """Bereken MSP en Entropy (en eventueel adjusted via Cleanlab)."""
    # Originele softmax
    msp = probs
    #print("probs", probs.shape) probs torch.Size([3334, 10]) -> ID dataset probs torch.Size([10000, 10]) -> OD dataset 
    if torch.isnan(msp).any():
        print("⚠️ Warning: NaNs detected in MSP scores, replacing with 0.")
        
    entropy = -(probs * probs.log()).sum(dim=1)
    if torch.isnan(entropy).any():
        print("⚠️ Warning: NaNs detected in entropy scores, replacing with 0.")
        entropy = torch.nan_to_num(entropy, nan=0.0)
    
    if c is not None:
        adj_probs = adjust_probs(probs, c)
        msp_adj = adj_probs.max(dim=1).values
        entropy_adj = -(adj_probs * adj_probs.log()).sum(dim=1)
        if torch.isnan(entropy_adj).any():
            print("⚠️ Warning: NaNs are stll detected in entropy_adj scores, replacing with 0.")
            entropy_adj = torch.nan_to_num(entropy_adj, nan=0.0)
    else:
        msp_adj = msp
        entropy_adj = entropy

    return {
        "msp": msp.cpu(),
        "entropy": entropy.cpu(),
        "msp_adj": msp_adj.cpu(),
        "entropy_adj": entropy_adj.cpu(),
    }

def analyze_checkpoint(ckpt_path, client_id, round_id, out_dir):
    # Model laden
    model = ClassifierCIFAR10.load_from_checkpoint(ckpt_path)
    model.eval()

    trainer = pl.Trainer(logger=False, enable_progress_bar=False)

    # Data opnieuw laden
    train_loader, _, id_loader = load_dataset(partition_id=int(client_id.replace("client","")), num_partitions=3, batch_size=32)
    ood_loader = load_dataset_ood(batch_size=32)

    # Compute confident thresholds c
    c = compute_confident_thresholds(model, train_loader, 10)

    # Test op ID
    trainer.test(model, id_loader)
    all_probs_id = torch.cat([x["probs"] for x in model.test_outputs if "probs" in x]).cpu()

    # Test op OOD
    trainer.test(model, ood_loader)
    all_probs_ood = torch.cat([x["probs_ood"] for x in model.test_outputs if "probs_ood" in x]).cpu()

    # Bereken OOD scores
    scores_id = compute_ood_scores(all_probs_id, c)
    scores_ood = compute_ood_scores(all_probs_ood, c)

    results = {}
    for key in ["msp", "msp_adj", "entropy", "entropy_adj"]:
        y_true = torch.cat([
            torch.zeros_like(scores_id[key], dtype=torch.int32),
            torch.ones_like(scores_ood[key], dtype=torch.int32),
        ])
        # Voor MSP geldt lage score = OOD → negatief
        if "msp" in key:
            y_score = torch.cat([scores_id[key], scores_ood[key]]) * -1
        else:
            y_score = torch.cat([scores_id[key], scores_ood[key]])

        auroc = roc_auc_score(y_true.numpy(), y_score.numpy())

        results[key] = auroc

    # ---- AUROC berekenen ----
    y_true = torch.cat([
        torch.zeros_like(all_probs_id, dtype=torch.int32),
        torch.ones_like(all_probs_ood, dtype=torch.int32),
    ])
    y_score = torch.cat([all_probs_id, all_probs_ood])
    auroc = roc_auc_score(y_true.numpy(), y_score.numpy())
    print(f"Client {client_id}, Round {round_id}, AUROC={auroc:.3f}")

    # # ---- Histogram ----
    # os.makedirs(out_dir, exist_ok=True)
    # plt.figure()
    # plt.hist(all_probs_id.numpy(), bins=50, alpha=0.6, label="ID")
    # plt.hist(all_probs_ood.numpy(), bins=50, alpha=0.6, label="OOD")
    # plt.xlabel("MSP")
    # plt.ylabel("Aantal samples")
    # plt.legend()
    # plt.title(f"MSP distributie (AUROC={auroc:.3f})")
    # plt.savefig(os.path.join(out_dir, f"ood_hist_{client_id}_round{round_id}.png"))
    # plt.close()

    # # ---- Confusion Matrix ----
    # preds = (y_score < 0.5).int()  # threshold 0.5
    # cm = confusion_matrix(y_true.numpy(), preds.numpy())
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ID", "OOD"])
    # disp.plot()
    # plt.title(f"Confusion Matrix ({client_id}, {round_id})")
    # plt.savefig(os.path.join(out_dir, f"confmat_{client_id}_round{round_id}.png"))
    # plt.close()

    return results

if __name__ == "__main__":
    base_dir = "ood_federated_learning"   # bin folder
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
            res = analyze_checkpoint(
                ckpt_path,
                client_id=client.split("_")[0],   # bv. "client0"
                round_id=round_dir,               # bv. "round1"
                out_dir=client_path
            )

            client_id = client.split("_")[0]
            res["client"] = client_id
            res["round"] = round_dir
            results.append(res)

            # results.append({
            #     "client": client,
            #     "round": round_dir,
            #     "auroc": auroc
            # })

    if results:
        df = pd.DataFrame(results)
        avg_auroc = df["auroc"].mean()
        print(f"\n Gemiddelde AUROC over alle clients/rondes: {avg_auroc:.4f}")

    df = pd.DataFrame(results)
    df_mean = df.groupby("client")[["msp","msp_adj","entropy","entropy_adj"]].mean().reset_index()
    print(df_mean.to_markdown(index=False))