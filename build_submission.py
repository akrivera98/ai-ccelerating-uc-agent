import inspect
from pathlib import Path
from src.models.registry import _MODEL_REGISTRY
from src.util import load_config

# Map of model names to their source modules and dependencies
MODEL_SOURCES = {
    'Simple_MLP': {
        'classes': ['SimpleMLP'],
        'files': ['src/models/simple_mlp.py'],
    },
    'MLP_with_rounding': {
        'classes': ['MLP_with_rounding', 'RoundModel', 'diffFloor', 'diffGumbelBinarize', 'thresholdBinarize'],
        'files': ['src/models/simple_mlp_round.py', 'src/models/round.py', 'src/models/ste.py'],
    },
}

def extract_class_source(file_path, class_name):
    """Extract a single class from a file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find class definition
    start_idx = None
    indent = None
    for i, line in enumerate(lines):
        if f'class {class_name}' in line:
            start_idx = i
            indent = len(line) - len(line.lstrip())
            break
    
    if start_idx is None:
        return None
    
    # Find end of class (next line with same or less indent, or EOF)
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith('#'):
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= indent:
                end_idx = i
                break
    
    return ''.join(lines[start_idx:end_idx])

def build_submission(results_path, output_file="create_submission.py"):
    """Generate submission script"""
    cfg = load_config(f"{results_path}/config.yaml")
    model_name = cfg.model.name
    
    if model_name not in MODEL_SOURCES:
        raise ValueError(f"Model '{model_name}' not in MODEL_SOURCES. Add it first.")
    
    sources = MODEL_SOURCES[model_name]
    
    # Extract all class definitions
    class_defs = []
    for file_path in sources['files']:
        for class_name in sources['classes']:
            if class_name in sources['classes']:
                source = extract_class_source(file_path, class_name)
                if source:
                    # Clean up: remove decorators and imports from src
                    lines = source.split('\n')
                    cleaned = []
                    for line in lines:
                        if '@register_model' in line or 'from src.models' in line or 'import src.models' in line:
                            continue
                        cleaned.append(line)
                    class_defs.append('\n'.join(cleaned))
    
    # Generate script
    script = f'''import torch
import torch.nn as nn
import pandas as pd
import dill
import yaml
import inspect

# Class definitions from src/models (extracted automatically)
{chr(10).join(class_defs)}

# Wrapper class
class model:
    def __init__(self, model, generator_names):
        self.model = model
        self.generator_names = generator_names

    def transform_features(self, features):
        df_profiles = features["Profiles"]
        df_init_conditions = features["Initial_Conditions"]
        demand = torch.tensor(df_profiles["demand"].values, dtype=torch.float32)
        wind = torch.tensor(df_profiles["wind"].values, dtype=torch.float32)
        solar = torch.tensor(df_profiles["solar"].values, dtype=torch.float32)
        gen_init_power = torch.tensor(df_init_conditions["initial_power"].values, dtype=torch.float32)
        gen_init_status = torch.tensor(df_init_conditions["initial_status"].values, dtype=torch.float32)
        x = torch.cat([demand, wind, solar, gen_init_power, gen_init_status], dim=0)
        return x

    def transform_predictions(self, predictions):
        status_array = predictions.cpu().numpy().reshape(72, 51)
        return pd.DataFrame(status_array, index=range(72), columns=self.generator_names)

    def predict(self, features):
        status = {{}}
        for instance_index in features.keys():
            x = self.transform_features(features[instance_index])
            with torch.no_grad():
                self.model.eval()
                pred = self.model(x)
            status[instance_index] = self.transform_predictions(pred)
        return status

def main():
    results_path = "{results_path}"
    with open(f"{{results_path}}/config.yaml") as f:
        raw_cfg = f.read().replace("!!python/object:src.util.Config", "").replace("tag:yaml.org,2002:python/object:src.util.Config", "")
        config_dict = yaml.safe_load(raw_cfg)
    
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, Config(v) if isinstance(v, dict) else v)
    
    cfg = Config(config_dict)
    
    # Build model (update class name and params as needed)
    sig = inspect.signature({sources['classes'][0]}.__init__)
    init_params = {{k: getattr(cfg.model, k) for k in sig.parameters.keys() if k != 'self' and hasattr(cfg.model, k)}}
    model_inst = {sources['classes'][0]}(**init_params)
    model_inst.load_state_dict(torch.load(f"{{results_path}}/simple_mlp_state.pt"))
    
    gen_names = pd.read_excel("data/Train_Data/instance_2021_Q1_1/Response_Variables.xlsx").columns[1:].tolist()
    wrapped = model(model=model_inst, generator_names=gen_names)
    
    with open("submission/model.dill", "wb") as f:
        dill.dump(wrapped, f, recurse=True)
    print("Done!")

if __name__ == "__main__":
    main()
'''
    
    with open(output_file, 'w') as f:
        f.write(script)
    print(f"Generated {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", default="create_submission_test.py")
    args = parser.parse_args()
    build_submission(args.results, args.output)