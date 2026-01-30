import torch
from src.models.round import ste_round 

@torch.no_grad()
def compute_classification_accuracy(model, dataloader, device):
    is_on_correct = 0
    is_on_total = 0
    is_charging_correct = 0
    is_charging_total = 0
    is_discharging_correct = 0
    is_discharging_total = 0

    for batch in dataloader:
        features = {k: v.to(device) for k, v in batch["features"].items()}
        targets = {k: v.to(device) for k, v in batch["target"].items()}

        outputs_dict = model(features)
        is_on_rounded = ste_round(outputs_dict["is_on"])

        # Is_on accuracy
        is_on_correct += (is_on_rounded == targets["is_on"]).sum().item()
        is_on_total += targets["is_on"].numel()

        # is_charging accuracy
        is_charging_correct += (outputs_dict["is_charging"] == targets["is_charging"]).sum().item()
        is_charging_total += targets["is_charging"].numel()

        # is_discharging accuracy
        is_discharging_correct += (outputs_dict["is_discharging"] == targets["is_discharging"]).sum().item()
        is_discharging_total += targets["is_discharging"].numel()

    is_on_accuracy = is_on_correct / is_on_total if is_on_total > 0 else 0
    is_charging_accuracy = is_charging_correct / is_charging_total if is_charging_total > 0 else 0
    is_discharging_accuracy = is_discharging_correct / is_discharging_total if is_discharging_total > 0 else 0

    return {
        "is_on_accuracy": is_on_accuracy,
        "is_charging_accuracy": is_charging_accuracy,
        "is_discharging_accuracy": is_discharging_accuracy,
    }







