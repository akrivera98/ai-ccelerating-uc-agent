import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import glob


class SimpleDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.features_files = glob.glob(
            os.path.join(data_dir, "*", "explanatory_variables.xlsx")
        )
        self.target_files = glob.glob(
            os.path.join(data_dir, "*", "Response_Variables.xlsx")
        )

    def __len__(self) -> int:
        assert len(self.features_files) == len(self.target_files), (
            "Number of feature files and target files must be the same."
        )
        return len(self.features_files)

    def _load_file(self, path_features: str, path_targets: str) -> dict:
        df_profiles = pd.read_excel(path_features, sheet_name="Profiles")
        df_init_conditions = pd.read_excel(
            path_features, sheet_name="Initial_Conditions"
        )
        df_targets = pd.read_excel(path_targets).iloc[:, 1:]

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

        profiles = torch.tensor(df_profiles.iloc[:, 1:].to_numpy(), dtype=torch.float32)         # (72, 3) if demand/wind/solar
        init_conds = torch.tensor(df_init_conditions.iloc[:, 1:].to_numpy(), dtype=torch.float32)           # (51, 2) e.g. (init_power, init_status)
        y = torch.tensor(df_targets.to_numpy(), dtype=torch.float32)       

        return {"features": {"profiles": profiles, "initial_conditions": init_conds}, "target": y}

    def __getitem__(self, idx: int) -> dict:
        path_features = self.features_files[idx]
        path_targets = self.target_files[idx]
        sample = self._load_file(path_features, path_targets)
        return sample
