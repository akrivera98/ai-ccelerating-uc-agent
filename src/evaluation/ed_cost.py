from src.models.round import ste_round
import torch

@torch.no_grad()
def compute_ed_cost(model, ed_layer, dataloader, device, highs_time_limit=None):
    model.eval()
    ed_layer.eval()

    per_instance_costs = []

    for batch in dataloader:
        features = {k: v.to(device) for k, v in batch["features"].items()}

        outputs = model(features)
        is_on = ste_round(outputs["is_on"])
        is_charging = outputs["is_charging"]
        is_discharging = outputs["is_discharging"]

        load = features["profiles"][:, :, 0]                  # (B,T)
        solar_max = features["profiles"][:, :, 2].unsqueeze(1) # (B,1,T)
        wind_max  = features["profiles"][:, :, 1].unsqueeze(1) # (B,1,T)

        ed_cost = ed_layer.objective(
            load, solar_max, wind_max, is_on, is_charging, is_discharging, highs_time_limit
        )

        ed_cost = ed_cost.reshape(-1).detach().cpu()
        per_instance_costs.append(ed_cost)

    costs = torch.cat(per_instance_costs, dim=0)  # (N,)

    return {
        "ed_cost_mean": costs.mean().item(),
        "ed_cost_median": costs.median().item(),
        "ed_cost_min": costs.min().item(),
        "ed_cost_max": costs.max().item(),
        "n_instances": int(costs.numel()),
    }