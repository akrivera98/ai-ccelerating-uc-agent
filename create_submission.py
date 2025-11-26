import torch
import torch.nn as nn
import pandas as pd
import dill
import yaml


# simple net architecture
class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int,
        final_activation: str = None,
    ):
        super().__init__()

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        if final_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        if not self.training:
            out = (out > 0.5).float()

        return out


# Wrapper class for submission model
class model:
    def __init__(self, model: nn.Module, generator_names: list[str]):
        self.model = model
        self.generator_names = generator_names

    def transform_features(self, features):
        import torch

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
        import torch

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
    results_path = "results/simple_no_round/20251114_104110"

    config_path = f"{results_path}/config.yaml"

    with open(config_path, "r") as f:
        raw = f.read()

    # Remove the python/object tags
    cleaned = raw.replace("!!python/object:__main__.Config", "")

    config = yaml.safe_load(cleaned)

    # Get model parameters
    model_params = config["model"]
    input_size = model_params["input_size"]
    hidden_size = model_params["hidden_size"]
    num_hidden_layers = model_params["num_hidden_layers"]
    output_size = model_params["output_size"]
    final_activation = model_params["final_activation"]

    # Instantiate the model
    simple_net = SimpleMLP(
        input_size=input_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        output_size=output_size,
        final_activation=final_activation,
    )

    simple_net.load_state_dict(torch.load(f"{results_path}/simple_mlp_state.pt"))

    # get generator names
    response_vars_filename = (
        "data/starting_kit/Train_Data/instance_2021_Q1_1/Response_Variables.xlsx"
    )
    gen_names = pd.read_excel(response_vars_filename).columns[1:].tolist()

    wrapped = model(model=simple_net, generator_names=gen_names)

    # try to predict on a sample input
    sample_input = {
        "instance_2021_Q1_1": {
            "Profiles": pd.read_excel(
                "data/starting_kit/Train_Data/instance_2021_Q1_1/explanatory_variables.xlsx",
                sheet_name="Profiles",
            ),
            "Initial_Conditions": pd.read_excel(
                "data/starting_kit/Train_Data/instance_2021_Q1_1/explanatory_variables.xlsx",
                sheet_name="Initial_Conditions",
            ),
        }
    }

    print(wrapped.model.__class__.__module__)
    print(wrapped.model.__class__.__name__)

    with open("submission/model.dill", "wb") as file:
        dill.dump(wrapped, file)

    with open("submission/model.dill", "rb") as f:
        new_model = dill.load(f)

    sample_output = new_model.predict(sample_input)

    with open("submission/prediction.dill", "wb") as file:
        dill.dump(sample_output, file)


if __name__ == "__main__":
    main()
