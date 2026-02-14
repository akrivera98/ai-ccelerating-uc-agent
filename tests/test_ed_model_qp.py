import torch
from src.models.ed_model_qp import EDModelQP
from src.ed_models.data_utils import create_data_dict
import pandas as pd
import json


def _get_binary_variables(output_path: str):
    with open(output_path, "r") as f:
        output_data = json.load(f)

    is_on = torch.zeros((len(output_data["Is on"]), 72))
    is_charging = torch.zeros((len(output_data["Storage discharging rates (MW)"]), 72))
    is_discharging = torch.zeros(
        (len(output_data["Storage discharging rates (MW)"]), 72)
    )

    for i, (gen, unit_status) in enumerate(
        sorted(output_data["Is on"].items(), key=lambda x: x[0])
    ):
        is_on[i, :] = torch.tensor(unit_status, dtype=torch.double)

    for i, (storage, charging_rates) in enumerate(
        sorted(output_data["Storage charging rates (MW)"].items(), key=lambda x: x[0])
    ):
        rates_tensor = torch.tensor(charging_rates, dtype=torch.double)
        is_charging[i, :] = (rates_tensor > 0).double()
    for i, (storage, discharging_rates) in enumerate(
        sorted(
            output_data["Storage discharging rates (MW)"].items(), key=lambda x: x[0]
        )
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


sample_path = "data/starting_kit_ai-uc_v2/Train_Data/instance_2021_Q1_1/InputData.json"
explanatory_vars_path = "data/starting_kit_ai-uc_v2/Train_Data/instance_2021_Q1_1/explanatory_variables.xlsx"
output_path = "data/starting_kit_ai-uc_v2/Train_Data/instance_2021_Q1_1/OutputData.json"
data_dict = create_data_dict(sample_path)

load, solar_max, wind_max = _get_explanatory_variables(explanatory_vars_path)
is_on, is_charging, is_discharging = _get_binary_variables(output_path)

ed_model_qp = EDModelQP(data_dict)

y = ed_model_qp(
    load, solar_max, wind_max, is_on, is_charging, is_discharging, verbose=True
)

pass
