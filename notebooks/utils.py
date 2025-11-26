import matplotlib.pyplot as plt
import torch


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    if val_losses:
        val_epochs, val_loss_values = zip(*val_losses)
        plt.plot(val_epochs, val_loss_values, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()


def compute_test_metrics(model, test_loader, device="cpu"):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            y = batch["target"].to(device)

            preds = model(x)  # already hard-rounded inside forward()
            all_preds.append(preds)
            all_targets.append(y)

    # Concatenate all batches
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Compute true positives, false positives, false negatives
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {"precision": precision, "recall": recall, "f1_score": f1}

def compute_per_timestep_metrics(model, test_loader, n_periods, n_gens, device="cpu"):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            y = batch["target"].to(device)  # [B, n_outputs]

            preds = model(x)  # [B, n_outputs], already hard-rounded
            all_preds.append(preds)
            all_targets.append(y)

    preds = torch.cat(all_preds, dim=0)    # [N, n_outputs]
    targets = torch.cat(all_targets, dim=0)

    # reshape to [N, T, G]
    N = preds.shape[0]
    preds = preds.view(N, n_periods, n_gens)
    targets = targets.view(N, n_periods, n_gens)

    # per-timestep accuracy: fraction of correct commitments across batch & gens
    correct = (preds == targets).float()           # [N, T, G]
    per_timestep_acc = correct.mean(dim=(0, 2))    # [T]

    # optional: per-timestep F1
    tp = ((preds == 1) & (targets == 1)).sum(dim=(0, 2)).float()  # [T]
    fp = ((preds == 1) & (targets == 0)).sum(dim=(0, 2)).float()  # [T]
    fn = ((preds == 0) & (targets == 1)).sum(dim=(0, 2)).float()  # [T]

    precision_t = tp / (tp + fp + 1e-8)            # [T]
    recall_t = tp / (tp + fn + 1e-8)               # [T]
    f1_t = 2 * precision_t * recall_t / (precision_t + recall_t + 1e-8)

    return {
        "per_timestep_accuracy": per_timestep_acc.cpu().numpy(),  # shape [T]
        "per_timestep_precision": precision_t.cpu().numpy(),
        "per_timestep_recall": recall_t.cpu().numpy(),
        "per_timestep_f1": f1_t.cpu().numpy(),
    }



def compute_single_instance_timestep_metrics(
    model,
    features,      # tensor [1, n_outputs] or whatever your model expects
    target,        # tensor [1, n_outputs]
    n_periods,
    n_gens,
    device="cpu",
):
    """
    Compute per-timestep metrics for a single instance.

    Returns:
        per_timestep_acc: [T] tensor with accuracy per hour
        per_timestep_precision: [T] tensor
        per_timestep_recall: [T] tensor
        per_timestep_f1: [T] tensor
        error_mask: [T, G] bool tensor where True = prediction != target
        preds_2d: [T, G] predictions
        targets_2d: [T, G] targets
    """
    model.eval()

    features = features.to(device)
    target = target.to(device)

    with torch.no_grad():
        preds = model(features)  # [1, n_outputs]

    # reshape to [T, G]
    preds = preds.view(n_periods, n_gens)
    targets = target.view(n_periods, n_gens)

    # correctness per bit
    correct = (preds == targets)        # [T, G]
    error_mask = ~correct               # [T, G]

    # per-timestep accuracy (across generators)
    per_timestep_acc = correct.float().mean(dim=1)   # [T]

    # per-timestep precision/recall/F1
    tp = ((preds == 1) & (targets == 1)).sum(dim=1).float()  # [T]
    fp = ((preds == 1) & (targets == 0)).sum(dim=1).float()  # [T]
    fn = ((preds == 0) & (targets == 1)).sum(dim=1).float()  # [T]

    precision_t = tp / (tp + fp + 1e-8)
    recall_t = tp / (tp + fn + 1e-8)
    f1_t = 2 * precision_t * recall_t / (precision_t + recall_t + 1e-8)

    return {
        "per_timestep_accuracy": per_timestep_acc.cpu(),
        "per_timestep_precision": precision_t.cpu(),
        "per_timestep_recall": recall_t.cpu(),
        "per_timestep_f1": f1_t.cpu(),
        "error_mask": error_mask.cpu(),     # [T, G]
        "preds_2d": preds.cpu(),            # [T, G]
        "targets_2d": targets.cpu(),        # [T, G]
    }

def get_instance_from_loader(test_loader, k):
    """
    Return features, target for the k-th instance in the loader.
    Assumes each batch is a dict with keys 'features' and 'target'.
    """
    seen = 0
    for batch in test_loader:
        x = batch["features"]      # [B, ...]
        y = batch["target"]        # [B, ...]
        B = x.shape[0]

        if seen + B > k:
            idx = k - seen
            return x[idx:idx+1], y[idx:idx+1]   # keep batch dim = 1

        seen += B

    raise IndexError("k out of range")
