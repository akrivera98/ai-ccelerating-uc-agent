from torch import nn
import torch


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

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        order = sorted(x.keys())
        x = torch.cat([x[k] for k in order], dim=-1)

        out = self.net(x)
        if not self.training:
            out = (out > 0.5).float()

        return out


class MLP(nn.Module):
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

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        profiles = x["profiles"].reshape(x["profiles"].shape[0], -1)
        init_conds = x["initial_conditions"].reshape(
            x["initial_conditions"].shape[0], -1
        )
        x = torch.cat([profiles, init_conds], dim=-1)
        out = self.net(x)
        if not self.training:
            out = (out > 0.5).float()

        return out
