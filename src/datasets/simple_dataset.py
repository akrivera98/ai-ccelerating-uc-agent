import json
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import glob
import gzip


class SimpleDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.features_files = glob.glob(
            os.path.join(data_dir, "*", "explanatory_variables.xlsx")
        )
        self.target_files = glob.glob(
            os.path.join(data_dir, "*", "response_variables.xlsx")
        )

        self.output_gz = glob.glob(os.path.join(data_dir, "*", "OutputData.json.gz"))

    def __len__(self) -> int:
        assert len(self.features_files) == len(self.target_files), (
            "Number of feature files and target files must be the same."
        )
        return len(self.features_files)

    def _load_file(self, path_features: str, path_targets: str, path_gz: str) -> dict:
        df_profiles = pd.read_excel(path_features, sheet_name="Profiles")
        is_charging, is_discharging = self._get_storage_decisions_from_gz(path_gz)
        df_init_conditions = pd.read_excel(
            path_features, sheet_name="Initial_Conditions"
        )
        df_targets = pd.read_excel(path_targets).iloc[:, 1:]

        # Sort
        sorted_gens = sorted(df_targets.columns)
        df_targets = df_targets[sorted_gens]

        # Build features tensor
        demand = torch.tensor(df_profiles["demand"].values, dtype=torch.float32)
        wind = torch.tensor(df_profiles["wind"].values, dtype=torch.float32)
        solar = torch.tensor(df_profiles["solar"].values, dtype=torch.float32)
        gen_init_power = torch.tensor(
            df_init_conditions["initial_power"].values, dtype=torch.float32
        )
        gen_init_status = torch.tensor(
            df_init_conditions["initial_status"].values, dtype=torch.float32
        )

        features = torch.cat(
            [demand, wind, solar, gen_init_power, gen_init_status], dim=0
        )  # dims (72 * 3 + 51 * 2, )

        # Build targets tensor
        target = torch.tensor(df_targets.to_numpy(), dtype=torch.float32).reshape(
            -1
        )  # dims (72 * 51, )

        profiles = torch.tensor(
            df_profiles.iloc[:, 1:].to_numpy(), dtype=torch.float32
        )  # (72, 3) if demand/wind/solar
        init_conds = torch.tensor(
            df_init_conditions.iloc[:, 1:].to_numpy(), dtype=torch.float32
        )  # (51, 2) e.g. (init_power, init_status)
        y = torch.tensor(df_targets.to_numpy(), dtype=torch.float32).T

        return {
            "features": {"profiles": profiles, "initial_conditions": init_conds},
            "target": {
                "thermal_commitment": y,
                "is_charging": is_charging,
                "is_discharging": is_discharging,
            },
        }

    def __getitem__(self, idx: int) -> dict:
        path_features = self.features_files[idx]
        path_targets = self.target_files[idx]
        path_output_gz = self.output_gz[idx]
        sample = self._load_file(path_features, path_targets, path_output_gz)
        return sample

    def _get_storage_decisions_from_gz(
        self, gz_path: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read OutputData.json if it exists; otherwise read OutputData.json.gz.
        """
        json_path = gz_path[:-3]  # strip ".gz"

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                output_data = json.load(f)
        else:
            with gzip.open(gz_path, "rt") as f:
                output_data = json.load(f)

        charging_dict = output_data["Storage charging rates (MW)"]
        discharging_dict = output_data["Storage discharging rates (MW)"]

        storage_names = sorted(charging_dict.keys())
        S = len(charging_dict)
        T = 72

        is_charging = torch.zeros((S, T), dtype=torch.float32)
        is_discharging = torch.zeros((S, T), dtype=torch.float32)

        for i, s_name in enumerate(storage_names):
            rates = charging_dict[s_name]
            rates_tensor = torch.tensor(rates, dtype=torch.float32)
            is_charging[i, :] = (rates_tensor > 0).double()

        for i, s_name in enumerate(storage_names):
            rates = discharging_dict[s_name]
            rates_tensor = torch.tensor(rates, dtype=torch.float32)
            is_discharging[i, :] = (rates_tensor > 0).double()

        assert torch.all(is_charging * is_discharging == 0), (
            "Storage units cannot charge and discharge simultaneously."
        )

        return is_charging, is_discharging
