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
