import gzip
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# ---- Extracted model definition ----
class TwoHeadMLP_Flex(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        T: int = 72,
        G: int = 51,
        S: int = 14,
        tau: float = 1.0,
        predict_storage: bool = True,
        # --- subset selection ---
        gen_idx: Optional[Sequence[int]] = None,
        gen_names: Optional[Sequence[str]] = None,
        gen_names_all: Optional[Sequence[str]] = None,  # full ordered list length G
        return_full_is_on: bool = True,
        # --- rounding ---
        round_thermal: str = "none",
        thermal_threshold: float = 0.5,
        round_storage: str = "gumbel",
    ):
        super().__init__()
        import torch as _torch

        self.torch = _torch
        self.T, self.G, self.S = T, G, S
        self.tau = tau
        self.predict_storage = predict_storage
        self.return_full_is_on = return_full_is_on
        self.round_thermal = round_thermal
        self.thermal_threshold = thermal_threshold
        self.round_storage = round_storage

        # ---- resolve generator subset ----
        if gen_idx is not None and gen_names is not None:
            raise ValueError("Provide only one of gen_idx or gen_names, not both.")

        if gen_names is not None:
            if gen_names_all is None:
                raise ValueError(
                    "If gen_names is provided, you must also provide gen_names_all."
                )
            name_to_idx = {name: i for i, name in enumerate(gen_names_all)}
            missing = [n for n in gen_names if n not in name_to_idx]
            if missing:
                raise ValueError(
                    f"gen_names contains unknown names: {missing[:5]} (showing up to 5)"
                )
            gen_idx = [name_to_idx[n] for n in gen_names]

        if gen_idx is None:
            gen_idx = list(range(G))

        self.register_buffer("gen_idx", self.torch.tensor(gen_idx))
        self.G_out = len(gen_idx)

        # ---- trunk ----
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)

        # ---- heads ----
        self.thermal_head = nn.Linear(hidden_size, T * self.G_out)
        self.storage_head = (
            nn.Linear(hidden_size, T * S * 3) if predict_storage else None
        )

    def _round_thermal(self, probs, logits):
        """
        probs/logits shape: (B, T, G_out)
        returns decisions shape: (B, T, G_out)
        """
        if self.round_thermal == "none":
            return probs

        if self.round_thermal == "threshold_ste":
            # STE only matters during training (for gradients).
            # During eval/inference we do a hard threshold (pickle-safe).
            if self.training:
                return ste_round(probs)
            else:
                return (probs > self.thermal_threshold).float()

        if self.round_thermal == "gumbel":  # TODO: check this
            # treat each on/off as a 2-class problem (off/on) per entry
            # logits2 shape: (B, T, G_out, 2)
            off_logits = self.torch.zeros_like(logits)
            logits2 = self.torch.stack([off_logits, logits], dim=-1)
            y = F.gumbel_softmax(logits2, tau=self.tau, hard=True, dim=-1)
            return y[..., 1]

        raise ValueError(f"Unknown round_thermal='{self.round_thermal}'")

    def _storage_outputs(self, h):
        """
        returns:
          storage_logits: (B, T, S, 3)
          is_charging:    (B, T, S)
          is_discharging: (B, T, S)
          storage_probs:  (B, T, S, 3)
        """
        storage_logits = self.storage_head(h).view(-1, self.T, self.S, 3)

        if self.round_storage == "none":
            # soft probabilities per class
            y = F.softmax(storage_logits, dim=-1)
        elif self.round_storage == "gumbel":
            y = F.gumbel_softmax(storage_logits, tau=self.tau, hard=True, dim=-1)
        else:
            raise ValueError(f"Unknown round_storage='{self.round_storage}'")

        return {
            "storage_logits": storage_logits,
            "is_charging": y[..., 1],
            "is_discharging": y[..., 2],
            "storage_probs": y,
        }

    def forward(self, x):
        B = x["profiles"].shape[0]

        # Expect profiles shaped like (B, T, ...) and initial_conditions (B, ...)
        profiles = x["profiles"].reshape(B, -1)
        init_conds = x["initial_conditions"].reshape(
            B, -1
        )  # TODO: check init conds shape
        feats = self.torch.cat([profiles, init_conds], dim=-1)

        h = self.trunk(feats)

        # ---- thermal ----
        thermal_logits = self.thermal_head(h).view(
            B, self.T, self.G_out
        )  # (B, T, G_out)
        thermal_probs = self.torch.sigmoid(thermal_logits)
        thermal_decisions = self._round_thermal(
            thermal_probs, thermal_logits
        )  # (B, T, G_out)

        # scatter back to full (B, T, G) if requested and subset is used
        if self.return_full_is_on and self.G_out != self.G:
            is_on_full = self.torch.zeros(
                B,
                self.T,
                self.G,
                device=thermal_decisions.device,
            )
            is_on_full[:, :, self.gen_idx] = thermal_decisions
            is_on_out = is_on_full
        else:
            is_on_out = thermal_decisions  # (B, T, G_out) or (B, T, G)

        out = {
            "is_on": is_on_out,
            "thermal_logits": thermal_logits,
            "thermal_probs": thermal_probs,
            "gen_idx": self.gen_idx,
        }

        # ---- storage (optional, time-first) ----
        if self.predict_storage:
            out.update(self._storage_outputs(h))

        return out


# ---- Wrapper class expected by the competition ----
class model:
    def __init__(self, model: nn.Module, generators: Dict[str, tuple], gen_order: list[str]):
        import torch as _torch
        self.torch = _torch
        self.model = model
        self.generators = generators
        self.generator_names = gen_order  # canonical order
        self.model.eval()

    def transform_features(self, features_one):
        torch = self.torch
        import numpy as np
        df_profiles = features_one["Profiles"]
        df_init = features_one["Initial_Conditions"]

        # profiles: (72,3) from col 1 onward
        prof_np = df_profiles.to_numpy(dtype=np.float32)
        profiles = torch.from_numpy(prof_np).unsqueeze(0)  # (1,72,3)

        # initial conditions: reorder gens to gen_order, then transpose to (1,2,51)
        # Assumes generator names are the index; if not, uncomment next line:
        # df_init = df_init.set_index(df_init.columns[0])

        df_init = df_init.loc[self.generator_names]
        init_np = df_init.to_numpy(dtype=np.float32)  # (51,2)
        init_conds = torch.from_numpy(init_np).T.unsqueeze(0)     # (1,2,51)

        return {"profiles": profiles, "initial_conditions": init_conds}

    def transform_predictions(self, is_on: torch.Tensor) -> pd.DataFrame:
        torch = self.torch
        arr = is_on.detach().cpu().numpy().reshape(72, 51)
        return pd.DataFrame(arr, index=range(72), columns=self.generator_names)

    def predict(self, features):
        torch = self.torch
        out = {}
        self.model.eval()
        with torch.no_grad():
            for instance_index in features.keys():
                x = self.transform_features(features[instance_index])
                pred = self.model(x)

                if isinstance(pred, dict) and "is_on" in pred:
                    is_on = pred["is_on"].squeeze(0)
                else:
                    is_on = pred.squeeze(0) if hasattr(pred, "ndim") and pred.ndim == 3 else pred

                df = self.transform_predictions(is_on)
                out[instance_index] = df
                # out[instance_index] = self.repair_feasibility(features[instance_index], df)
        return out


def main():
    results_path = "results/testing_scoring/20260205_222033"

    # ---- load clean config ----
    with open(f"{results_path}/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    hp = cfg["model"]["hyper_params"]

    # ---- instantiate + load weights ----
    m = TwoHeadMLP_Flex(**hp)
    sd = torch.load(f"{results_path}/simple_mlp_state.pt", map_location="cpu")
    m.load_state_dict(sd)
    m.eval()

    # ---- generator metadata + canonical order ----
    with gzip.open("data/Train_Data/instance_2021_Q1_1/InputData.json.gz", "r") as f:
        data = json.loads(f.read().decode("utf-8"))

    # Use JSON generator order as canonical (you said you want JSON order)
    generators_all = data["Generators"]
    gen_order = list(generators_all.keys())

    # If you need to restrict to the 51 thermal generators used in scoring,
    # filter using Response_Variables.xlsx:
    resp_cols = pd.read_excel("data/Train_Data/instance_2021_Q1_1/Response_Variables.xlsx").columns[1:].tolist()
    resp_set = set(resp_cols)
    gen_order = sorted([g for g in gen_order if g in resp_set])

    generators = {
        g: (generators_all[g]["Minimum downtime (h)"], generators_all[g]["Minimum uptime (h)"])
        for g in gen_order
    }

    wrapped = model(model=m, generators=generators, gen_order=gen_order)

    Path("submission").mkdir(parents=True, exist_ok=True)
    with open("submission/model.dill", "wb") as f:
        dill.dump(wrapped, f)

if __name__ == "__main__":
    main()
