import torch
from src.models.ed_model import UCModel
from src.models.data_classes import create_data_dict
import json
import numpy as np


def _get_binary_variables(output_path: str):
    with open(output_path, "r") as f:
        output_data = json.load(f)

    is_on = torch.zeros((len(output_data["Is on"]), 72))
    is_charging = torch.zeros((len(output_data["Storage discharging rates (MW)"]), 72))
    is_discharging = torch.zeros(
        (len(output_data["Storage discharging rates (MW)"]), 72)
    )

    for i, (gen, unit_status) in enumerate(output_data["Is on"].items()):
        is_on[i, :] = torch.tensor(unit_status, dtype=torch.double)

    for i, (storage, charging_rates) in enumerate(
        output_data["Storage charging rates (MW)"].items()
    ):
        rates_tensor = torch.tensor(charging_rates, dtype=torch.double)
        is_charging[i, :] = (rates_tensor > 0).double()
    for i, (storage, discharging_rates) in enumerate(
        output_data["Storage discharging rates (MW)"].items()
    ):
        rates_tensor = torch.tensor(discharging_rates, dtype=torch.double)
        is_discharging[i, :] = (rates_tensor > 0).double()

    assert torch.all(is_charging * is_discharging == 0), (
        "Storage units cannot charge and discharge simultaneously."
    )

    return is_on, is_charging, is_discharging


sample_path = "data/starting_kit/Train_Data/instance_2021_Q1_1/InputData.json"
output_path = "data/starting_kit/Train_Data/instance_2021_Q1_1/OutputData.json"

data_dict = create_data_dict(sample_path)
is_on_t, is_charging_t, is_discharging_t = _get_binary_variables(output_path)


uc_model = UCModel(data_dict)
uc_model.build()

# Set parameters
uc_model.thermal_gens.is_on.value = is_on_t.cpu().numpy()
uc_model.storage_units.is_charging.value = is_charging_t.cpu().numpy()
uc_model.storage_units.is_discharging.value = is_discharging_t.cpu().numpy()

solution = uc_model.solve(verbose=True)

print("Status:", uc_model.problem.status)
for c in uc_model.problem.constraints:
    v = c.violation()
    if v is not None and np.max(v) > 1e-4:
        print("Constraint violated by", np.max(v))
        print("  ", c)


# layer = uc_model.build_layer()

# T = uc_model.T
# num_units = uc_model.thermal_gens.num_units
# num_storage = uc_model.storage_units.num_units

# # Relaxed 0/1 as floats
# solution = layer(is_on_t, is_charging_t, is_discharging_t)

# # unpack if you want
# solution_dict = {name: sol for name, sol in zip(uc_model.variables.keys(), solution)}

# loss = solution_dict["curtailment"].sum()  # ez example
# loss.backward()
