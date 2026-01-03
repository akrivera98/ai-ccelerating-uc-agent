import torch
from src.models.ed_model import UCModel
from src.models.data_classes import create_data_dict
import json
import pickle
import pandas as pd


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

    return is_on.clamp_min(0), is_charging.clamp_min(0), is_discharging.clamp_min(0)


def _get_explanatory_variables(path_features: str):
    df_profiles = pd.read_excel(path_features, sheet_name="Profiles")
    load = torch.tensor(df_profiles["demand"].values, dtype=torch.float32)
    wind_max = torch.tensor(df_profiles["wind"].values, dtype=torch.float32)
    solar_max = torch.tensor(df_profiles["solar"].values, dtype=torch.float32)

    return load, solar_max, wind_max


sample_path = "data/starting_kit/Train_Data/instance_2021_Q1_1/InputData.json"
output_path = "data/starting_kit/Train_Data/instance_2021_Q1_1/OutputData.json"
cvs_path = "data/starting_kit/Train_Data/instance_2021_Q1_1/explanatory_variables.xlsx"

data_dict = create_data_dict(sample_path)
is_on_t, is_charging_t, is_discharging_t = _get_binary_variables(output_path)
load, solar_max, wind_max = _get_explanatory_variables(cvs_path)

uc_model = UCModel(data_dict)


# uc_model.build()

# # Set parameters
# uc_model.system.load.value = load.cpu().numpy()
# uc_model.profiled_gens.max_power_solar.value = solar_max.unsqueeze(0).cpu().numpy()
# uc_model.profiled_gens.max_power_wind.value = wind_max.unsqueeze(0).cpu().numpy()
# uc_model.thermal_gens.is_on.value = is_on_t.cpu().numpy()
# uc_model.storage_units.is_charging.value = is_charging_t.cpu().numpy()
# uc_model.storage_units.is_discharging.value = is_discharging_t.cpu().numpy()

# obj_val = uc_model.solve(verbose=True)

# solution_dict = {name: sol.value for name, sol in uc_model.variables.items()}
# solution_dict["objective_value"] = obj_val
# solution_dict["pmin"] = uc_model.thermal_gens.min_power

# with open("ed_solution_again.pkl", "wb") as f:
#     pickle.dump(solution_dict, f)


layer = uc_model.build_layer()

T = uc_model.T
num_units = uc_model.thermal_gens.num_units
num_storage = uc_model.storage_units.num_units

# Relaxed 0/1 as floats
solution = layer(
    load.to(torch.double),  # (72,)
    solar_max.unsqueeze(0).to(torch.double),  # (1,72)
    wind_max.unsqueeze(0).to(torch.double),  # (1,72)
    is_on_t.to(torch.double),  # (G,72)
    is_charging_t.to(torch.double),  # (S,72)
    is_discharging_t.to(torch.double),  # (S,72)
    solver_args={
        "eps": 1e-4,
        "max_iters": 100000,
        "acceleration_lookback": 0,
        "verbose": True,
    },
)

solution_dict = {name: sol for name, sol in zip(uc_model.variables.keys(), solution)}
solution_dict["pmin"] = uc_model.thermal_gens.min_power

# with open("ed_solution_layer.pkl", "wb") as f:
#     pickle.dump(solution_dict, f)
