# build_submission.py
import argparse
from pathlib import Path
import yaml


# -------------------------------
# Configure these two for your repo
# -------------------------------
MODEL_NAME = "TwoHeadMLP_Flex"
MODEL_FILE = "src/models/fnn.py"  # <-- CHANGE if your file is fnn.py
WEIGHTS_FILE = "simple_mlp_state.pt"  # <-- CHANGE if your weights file name differs


def extract_class_source(file_path: str, class_name: str) -> str | None:
    """Extract a single class definition block by indentation."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    start_idx = None
    indent = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(f"class {class_name}"):
            start_idx = i
            indent = len(line) - len(line.lstrip())
            break

    if start_idx is None:
        return None

    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        ln = lines[j]
        if ln.strip() and not ln.strip().startswith("#"):
            ln_indent = len(ln) - len(ln.lstrip())
            if ln_indent <= indent:
                end_idx = j
                break

    return "".join(lines[start_idx:end_idx])


def clean_source(src: str) -> str:
    """
    Remove repo-only decorators/imports that will break inside create_submission.py.
    Keep this conservative.
    """
    cleaned = []
    for line in src.splitlines():
        if "@register_model" in line:
            continue
        if line.strip().startswith("from src.") or line.strip().startswith(
            "import src."
        ):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def build_submission(results_path: str, output_file: str = "create_submission.py"):
    # --- load clean config.yaml ---
    cfg_path = Path(results_path) / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.yaml at: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    if model_name != MODEL_NAME:
        raise ValueError(
            f"Expected model.name='{MODEL_NAME}' but got '{model_name}'. "
            f"Edit MODEL_NAME / MODEL_FILE accordingly."
        )

    # --- extract model class ---
    src = extract_class_source(MODEL_FILE, MODEL_NAME)
    if not src:
        raise ValueError(f"Could not find class '{MODEL_NAME}' in {MODEL_FILE}")

    model_def = clean_source(src)

    # --- generate create_submission.py ---
    script = f'''import gzip
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
{model_def}


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

        return {{"profiles": profiles, "initial_conditions": init_conds}}

    def transform_predictions(self, is_on: torch.Tensor) -> pd.DataFrame:
        torch = self.torch
        arr = is_on.detach().cpu().numpy().reshape(72, 51)
        return pd.DataFrame(arr, index=range(72), columns=self.generator_names)

    def predict(self, features):
        torch = self.torch
        out = {{}}
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
    results_path = "{results_path}"

    # ---- load clean config ----
    with open(f"{{results_path}}/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    hp = cfg["model"]["hyper_params"]

    # ---- instantiate + load weights ----
    m = {MODEL_NAME}(**hp)
    sd = torch.load(f"{{results_path}}/{WEIGHTS_FILE}", map_location="cpu")
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

    generators = {{
        g: (generators_all[g]["Minimum downtime (h)"], generators_all[g]["Minimum uptime (h)"])
        for g in gen_order
    }}

    wrapped = model(model=m, generators=generators, gen_order=gen_order)

    Path("submission").mkdir(parents=True, exist_ok=True)
    with open("submission/model.dill", "wb") as f:
        dill.dump(wrapped, f)

if __name__ == "__main__":
    main()
'''

    with open(output_file, "w") as f:
        f.write(script)

    print(f"Generated {output_file} from {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="create_submission.py")
    args = parser.parse_args()
    build_submission(args.results, args.output)
