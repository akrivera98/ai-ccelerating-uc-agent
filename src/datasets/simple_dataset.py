import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import os
import glob
import gzip
from src.registry import registry


@registry.register_dataset("simple_dataset")
class SimpleDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir: str,
        fixed_off_gens=None,
        fixed_on_gens=None,
        include_profiled_utilization: bool = False,
    ):
        self.data_dir = data_dir
        self.fixed_off_gens = list(fixed_off_gens or [])
        self.fixed_on_gens = list(fixed_on_gens or [])
        self.include_profiled_utilization = include_profiled_utilization

        features = glob.glob(os.path.join(data_dir, "*", "explanatory_variables.xlsx"))
        targets = glob.glob(os.path.join(data_dir, "*", "response_variables.xlsx"))
        outputs = glob.glob(os.path.join(data_dir, "*", "OutputData.json.gz"))
        inputs = glob.glob(os.path.join(data_dir, "*", "InputData.json.gz"))

        # Key everything by instance directory
        def key(p: str) -> str:
            return os.path.dirname(p)

        feat_map = {key(p): p for p in features}
        targ_map = {key(p): p for p in targets}
        out_map = {key(p): p for p in outputs}
        in_map = {key(p): p for p in inputs}

        # Instance directories present in all three
        instance_dirs = sorted(feat_map.keys())
        self.instance_dirs = instance_dirs

        # Rebuild aligned lists
        self.features_files = [feat_map[d] for d in instance_dirs]
        self.target_files = [targ_map[d] for d in instance_dirs]
        self.output_gz = [out_map[d] for d in instance_dirs]
        self.input_gz = [in_map[d] for d in instance_dirs]

        self.gen_names_all, self.fixed_off_idx, self.fixed_on_idx, self.pred_idx = (
            self._build_generator_partitions()
        )
        self.lp_gen_idx = torch.cat([self.pred_idx, self.fixed_on_idx]).unique(
            sorted=True
        )  # gens included in the LP formulation

        # Final invariant check
        assert (
            len(self.features_files) == len(self.target_files) == len(self.output_gz)
        ), "Mismatch after alignment."

    def _build_generator_partitions(self):
        # cannonical ordering from the first instance
        df_targets = pd.read_excel(self.target_files[0]).iloc[:, 1:]
        all_sorted_gen_names = sorted(df_targets.columns)
        G = len(all_sorted_gen_names)

        name_to_idx = {name: i for i, name in enumerate(all_sorted_gen_names)}

        # validate profied names
        missing_off = [n for n in self.fixed_off_gens if n not in all_sorted_gen_names]
        missing_on = [n for n in self.fixed_on_gens if n not in all_sorted_gen_names]

        if missing_off or missing_on:
            raise ValueError(
                f"Fixed off gens not in data: {missing_off}, fixed on gens not in data: {missing_on}"
            )

        fixed_off_idx = torch.tensor(
            [name_to_idx[n] for n in self.fixed_off_gens], dtype=torch.long
        )
        fixed_on_idx = torch.tensor(
            [name_to_idx[n] for n in self.fixed_on_gens], dtype=torch.long
        )

        # check overlap
        if set(fixed_off_idx.tolist()) & set(fixed_on_idx.tolist()):
            raise ValueError("Some gens cannot be both fixed on and fixed off.")

        # set as predicted everything else
        pred_mask = torch.ones(G, dtype=torch.bool)
        pred_mask[fixed_off_idx] = False
        pred_mask[fixed_on_idx] = False
        pred_idx = torch.arange(G, dtype=torch.long)[pred_mask]

        return all_sorted_gen_names, fixed_off_idx, fixed_on_idx, pred_idx

    def __len__(self) -> int:
        assert len(self.features_files) == len(self.target_files), (
            "Number of feature files and target files must be the same."
        )
        return len(self.features_files)

    def _load_file(self, path_features: str, path_targets: str, path_gz: str) -> dict:
        df_profiles = pd.read_excel(path_features, sheet_name="Profiles")
        storage_status, charge_rates, discharge_rates, sorted_storage = (
            self._get_storage_decisions_from_gz(path_gz)
        )
        df_init_conditions = pd.read_excel(
            path_features, sheet_name="Initial_Conditions"
        )
        df_targets = pd.read_excel(path_targets).iloc[:, 1:]

        # Sort
        sorted_gens = self.gen_names_all
        df_targets = df_targets[sorted_gens]
        df_init_conditions = df_init_conditions.set_index(df_init_conditions.columns[0])
        df_init_conditions = df_init_conditions.loc[sorted_gens]

        # Build features tensors
        profiles = torch.tensor(
            df_profiles.iloc[:, 1:].to_numpy(), dtype=torch.float32
        )  # (72, 3) demand/wind/solar
        init_conds = torch.tensor(
            df_init_conditions.to_numpy(), dtype=torch.float32
        ).T  # (2, 51) e.g. (init_power, init_status)
        y = torch.tensor(df_targets.to_numpy(), dtype=torch.float32)

        return {
            "features": {"profiles": profiles, "initial_conditions": init_conds},
            "target": {
                "is_on": y,
                "storage_status": storage_status,  # charging, discharging, idle
            },
            "gen_names": sorted_gens,
            "storage_names": sorted_storage,
            "charge_rates": charge_rates,
            "discharge_rates": discharge_rates,
            # "profiled_gen_utilization": self._get_profiled_utilization(
            #     path_gz, profiles
            # ) if self.include_profiled_utilization else None,
        }

    def __getitem__(self, idx: int) -> dict:
        path_features = self.features_files[idx]
        path_targets = self.target_files[idx]
        path_output_gz = self.output_gz[idx]
        sample = self._load_file(path_features, path_targets, path_output_gz)
        return sample

    def _get_storage_decisions_from_gz(self, gz_path: str) -> torch.Tensor:
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

        storage_status = torch.zeros(
            (T, S, 3), dtype=torch.int64
        )  # charging, discharging, idle

        all_charge_rates = torch.zeros((T, S))
        all_discharge_rates = torch.zeros((T, S))

        for i, s_name in enumerate(storage_names):
            charge_rates = torch.tensor(charging_dict[s_name], dtype=torch.float32)
            discharge_rates = torch.tensor(
                discharging_dict[s_name], dtype=torch.float32
            )

            storage_status[:, i, 0] = (charge_rates > EPS).int()
            storage_status[:, i, 1] = (discharge_rates > EPS).int()

            storage_status[:, i, 2] = (
                1 - storage_status[:, i, 0] - storage_status[:, i, 1]
            )

            all_charge_rates[:, i] = charge_rates
            all_discharge_rates[:, i] = discharge_rates

        assert torch.all(storage_status[:, :, 0] * storage_status[:, :, 1] == 0), (
            "Storage units cannot charge and discharge simultaneously."
        )

        return storage_status, all_charge_rates, all_discharge_rates, storage_names

    def _get_profiled_utilization(self, path_gz, profiles):
        json_path = path_gz[:-3]  # strip ".gz"

        input_json_path = self.input_gz[0][
            :-3
        ]  # you only need to read one input file to problem parameters

        if os.path.exists(input_json_path):
            with open(input_json_path, "r") as f:
                input_data = json.load(f)
        else:
            with gzip.open(self.input_gz[0], "rt") as f:
                input_data = json.load(f)

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                output_data = json.load(f)
        else:
            with gzip.open(path_gz, "rt") as f:
                output_data = json.load(f)

        profiled_units_names = sorted(output_data["Profiled production (MW)"].keys())
        profiled_generation = output_data["Profiled production (MW)"]
        profiled_generation = torch.tensor(
            [profiled_generation[gen] for gen in profiled_units_names],
            dtype=torch.float32,
        ).T  # (72, num_profiled_gens)

        name_to_max = {
            gen: gen_data["Maximum power (MW)"]
            for gen, gen_data in input_data["Generators"].items()
            if gen in profiled_units_names
        }

        max_cols = []
        T = 72
        for name in profiled_units_names:
            if name in {"solar", "wind"}:
                col = profiles[:, 1] if name == "wind" else profiles[:, 2]
            else:
                mx = name_to_max[name]

                if isinstance(mx, (int, float)):
                    # scalar -> expand to length T
                    col = torch.full((T,), float(mx), dtype=torch.float32)
                else:
                    # list/array -> tensor, must be length T
                    col = torch.tensor(mx, dtype=torch.float32)
                    if col.numel() != T:
                        raise ValueError(
                            f"Max power for {name} has length {col.numel()}, expected {T}"
                        )
            max_cols.append(col)

        # stack columns -> (T, G)
        profiled_max = torch.stack(max_cols, dim=1)

        # safe divide
        profiled_max = torch.clamp(profiled_max, min=1e-6)
        utilization = profiled_generation / profiled_max
        return utilization
