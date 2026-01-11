import torch
import torch.nn as nn
import pandas as pd
import dill
import yaml
import inspect

# Class definitions from src/models (extracted automatically)
class MLP_with_rounding(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int,
        rounding_strategy: str,
    ):
        super().__init__()
        self.rounding_strategy = rounding_strategy

        ff_layers = []
        round_layers = []
        # Input layer
        ff_layers.append(nn.Linear(input_size, hidden_size))
        ff_layers.append(nn.ReLU())
        round_layers.append(nn.Linear(input_size + output_size, hidden_size))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            ff_layers.append(nn.Linear(hidden_size, hidden_size))
            ff_layers.append(nn.ReLU())
            round_layers.append(nn.Linear(hidden_size, hidden_size))
            round_layers.append(nn.ReLU())

        # Output layer
        ff_output_layer = nn.Linear(hidden_size, output_size)
        round_layers.append(nn.Linear(hidden_size, output_size))
        ff_layers.append(ff_output_layer)
        # Remove for MSELoss
        # ff_layers.append(nn.Sigmoid())


        self.ff_net = nn.Sequential(*ff_layers)
        self.round_net = RoundModel(nn.Sequential(*round_layers), rounding_strategy=rounding_strategy)
        
        # Initialize final layer with smaller weights to prevent sigmoid saturation
        # This keeps outputs in a learnable range (not too close to 0 or 1)
        nn.init.xavier_uniform_(ff_output_layer.weight, gain=0.1)  # Small gain prevents saturation
        nn.init.zeros_(ff_output_layer.bias)

    def forward(self, x):
        p = self.ff_net(x)
        return self.round_net(p, x)
    
class RoundModel(nn.Module):
    """
    Learnable model to round integer variables
    """
    def __init__(self, net, rounding_strategy="", tolerance=1e-3):
        super(RoundModel, self).__init__()
        # numerical tolerance
        self.tolerance = tolerance
        self.round_strat = rounding_strategy
        # autograd functions
        self.floor = diffFloor()
        if self.round_strat == "RC":
            self.bin = diffGumbelBinarize(temperature=1.0)
        elif self.round_strat == "LT":
            self.bin = thresholdBinarize()
        else:
            raise ValueError(f"Unknown rounding strategy: {rounding_strategy}")
        # sequence
        self.net = net

    def forward(self, sol, x):
        # concatenate all features: sol + features
        f = torch.cat([x, sol], dim=-1)
        # forward
        h = self.net(f)
        # rounding
    
        sol_hat = self.floor(sol)
        mask = torch.isclose(sol, torch.tensor(0.0), rtol=1e-4) | \
            torch.isclose(sol, torch.tensor(1.0), rtol=1e-4)
        if self.round_strat == "RC":
            v = self.bin(h)[~mask]
        elif self.round_strat == "LT":
            r = (sol[~mask] - sol_hat[~mask]) - h[~mask]
            v = self.bin(r)
        else:
            raise ValueError(f"Unknown rounding strategy: {self.round_strat}")
        sol_hat[~mask] += v
        return torch.clamp(sol_hat, 0.0, 1.0)

    def freeze(self):
        """
        Freezes the parameters of the callable in this node
        """
        for param in self.layers.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        Unfreezes the parameters of the callable in this node
        """
        for param in self.layers.parameters():
            param.requires_grad = True
class diffFloor(nn.Module):
    """
    An autograd model to floor numbers that applies a straight-through estimator
    for the backward pass.
    """
    def __init__(self):
        super(diffFloor, self).__init__()

    def forward(self, x):
        # floor
        x_floor = torch.floor(x).float()
        # apply the STE trick to keep the gradient
        return x_floor + (x - x.detach())


class diffGumbelBinarize(nn.Module):
    """
    An autograd function to binarize numbers using the Gumbel-Softmax trick,
    allowing gradients to be backpropagated through discrete variables.
    """
    def __init__(self, temperature=1.0, eps=1e-9):
        super(diffGumbelBinarize, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, x):
        # train mode
        if self.training:
            # Gumbel sampling
            gumbel_noise0 = self._gumbelSample(x)
            gumbel_noise1 = self._gumbelSample(x)
            # sigmoid with Gumbel
            noisy_diff = x + gumbel_noise1 - gumbel_noise0
            soft_sample = torch.sigmoid(noisy_diff / self.temperature)
            # hard rounding
            hard_sample = (soft_sample > 0.5).float()
            # apply the STE trick to keep the gradient
            return hard_sample + (soft_sample - soft_sample.detach())
        # eval mode
        else:
            # use a temperature-scaled sigmoid in evaluation mode for consistency
            return (torch.sigmoid(x / self.temperature) > 0.5).float()

    def _gumbelSample(self, x):
        """
        Generates Gumbel noise based on the input shape and device
        """
        u = torch.rand_like(x)
        return - torch.log(- torch.log(u + self.eps) + self.eps)



class thresholdBinarize(nn.Module):
    """
    An autograd function smoothly rounds the elements in `x` based on the
    corresponding values in `threshold` using a sigmoid function.
    """
    def __init__(self, threshold=0.5, slope=10):
        super(thresholdBinarize, self).__init__()
        self.slope = slope
        self.threshold = torch.tensor([threshold])

    def forward(self, x):
        # ensure the threshold_tensor values are between 0 and 1
        threshold = torch.clamp(self.threshold, 0, 1)
        # hard rounding
        hard_round = (x >= self.threshold).float()
        # calculate the difference and apply the sigmoid function
        diff = x - threshold
        smoothed_round = torch.sigmoid(self.slope * diff)
        # apply the STE trick to keep the gradient
        return hard_round + (smoothed_round - smoothed_round.detach())




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
        status = {}
        for instance_index in features.keys():
            x = self.transform_features(features[instance_index])
            with torch.no_grad():
                self.model.eval()
                pred = self.model(x)
            status[instance_index] = self.transform_predictions(pred)
        return status

def main():
    results_path = "results/simple_round/20260110_234202"
    with open(f"{results_path}/config.yaml") as f:
        raw_cfg = f.read().replace("!!python/object:src.util.Config", "").replace("tag:yaml.org,2002:python/object:src.util.Config", "")
        config_dict = yaml.safe_load(raw_cfg)
    
    class Config:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, Config(v) if isinstance(v, dict) else v)
    
    cfg = Config(config_dict)
    
    # Build model (update class name and params as needed)
    sig = inspect.signature(MLP_with_rounding.__init__)
    init_params = {k: getattr(cfg.model, k) for k in sig.parameters.keys() if k != 'self' and hasattr(cfg.model, k)}
    model_inst = MLP_with_rounding(**init_params)
    model_inst.load_state_dict(torch.load(f"{results_path}/simple_mlp_state.pt"))
    
    gen_names = pd.read_excel("data/Train_Data/instance_2021_Q1_1/Response_Variables.xlsx").columns[1:].tolist()
    wrapped = model(model=model_inst, generator_names=gen_names)
    
    with open("submission/model.dill", "wb") as f:
        dill.dump(wrapped, f, recurse=True)
    print("Done!")

if __name__ == "__main__":
    main()
