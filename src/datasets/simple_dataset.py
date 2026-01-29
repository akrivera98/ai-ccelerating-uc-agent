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

        features = glob.glob(os.path.join(data_dir, "*", "explanatory_variables.xlsx"))
        targets = glob.glob(os.path.join(data_dir, "*", "response_variables.xlsx"))
        outputs = glob.glob(os.path.join(data_dir, "*", "OutputData.json.gz"))

        # Key everything by instance directory
        def key(p: str) -> str:
            return os.path.dirname(p)

        feat_map = {key(p): p for p in features}
        targ_map = {key(p): p for p in targets}
        out_map = {key(p): p for p in outputs}

        # Instance directories present in all three
        instance_dirs = sorted(feat_map.keys())

        # Rebuild aligned lists
        self.features_files = [feat_map[d] for d in instance_dirs]
        self.target_files = [targ_map[d] for d in instance_dirs]
        self.output_gz = [out_map[d] for d in instance_dirs]

        # Final invariant check
        assert (
            len(self.features_files) == len(self.target_files) == len(self.output_gz)
        ), "Mismatch after alignment."

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
                "is_on": y,
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
        EPS = 1e-3  # small threshold to determine if charging/discharging

        is_charging = torch.zeros((S, T), dtype=torch.float32)
        is_discharging = torch.zeros((S, T), dtype=torch.float32)

        for i, s_name in enumerate(storage_names):
            charge_rates = torch.tensor(charging_dict[s_name], dtype=torch.float32)
            discharge_rates = torch.tensor(
                discharging_dict[s_name], dtype=torch.float32
            )

            is_charging[i, :] = (charge_rates > EPS).float()
            is_discharging[i, :] = (discharge_rates > EPS).float()
        assert torch.all(is_charging * is_discharging == 0), (
            "Storage units cannot charge and discharge simultaneously."
        )

        return is_charging, is_discharging
