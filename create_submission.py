import torch
import torch.nn as nn
import pandas as pd
import dill
from src.models.registry import build_model
from src.util import load_config


# Wrapper class for submission model
class model:
    def __init__(self, model: nn.Module, generator_names: list[str]):
        self.model = model
        self.generator_names = generator_names

    def transform_features(self, features):
        # Convert features dict of DataFrame to tensor
        df_profiles = features["Profiles"]
        df_init_conditions = features["Initial_Conditions"]
        demand = torch.tensor(df_profiles["demand"].values, dtype=torch.float32)
        wind = torch.tensor(df_profiles["wind"].values, dtype=torch.float32)
        solar = torch.tensor(df_profiles["solar"].values, dtype=torch.float32)
        gen_init_power = torch.tensor(
            df_init_conditions["initial_power"].values, dtype=torch.float32
        )
        gen_init_status = torch.tensor(
            df_init_conditions["initial_status"].values, dtype=torch.float32
        )

        x = torch.cat([demand, wind, solar, gen_init_power, gen_init_status], dim=0)
        return x

    def transform_predictions(self, predictions) -> pd.DataFrame:
        # Convert predictions to DataFrame

        status_array = predictions.cpu().numpy().reshape(72, 51)

        status = pd.DataFrame(
            status_array, index=range(72), columns=self.generator_names
        )
        return status

    def predict(self, features) -> dict[str, pd.DataFrame]:
        status = {}
        for instance_index in features.keys():
            x = self.transform_features(features[instance_index])
            with torch.no_grad():
                self.model.eval()
                pred = self.model(x)
            status[instance_index] = self.transform_predictions(pred)

        return status


def main():
    # results path
    results_path = "results/simple_round/20251210_001605"

    config_path = f"{results_path}/config.yaml"
    cfg = load_config(config_path)

    # Instantiate the model
    model_inst = build_model(cfg.model)
    model_inst.load_state_dict(torch.load(f"{results_path}/simple_mlp_state.pt"))

    # get generator names
    response_vars_filename = (
        "data/Train_Data/instance_2021_Q1_1/Response_Variables.xlsx"
    )
    gen_names = pd.read_excel(response_vars_filename).columns[1:].tolist()
    wrapped = model(model=model_inst, generator_names=gen_names)

    print(wrapped.model.__class__.__module__)
    print(wrapped.model.__class__.__name__)

    with open("submission/model.dill", "wb") as file:
        dill.dump(wrapped, file)



if __name__ == "__main__":
    main()
