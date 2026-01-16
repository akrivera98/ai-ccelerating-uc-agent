from torch import nn
import torch
import torch.nn.functional as F


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


class TwoHeadMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        T: int = 72,
        G: int = 51,
        S: int = 14,
        tau: float = 1.0,
        final_activation: str = None,
    ):
        super().__init__()
        self.T, self.G, self.S = T, G, S
        self.tau = tau

        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)

        # Heads
        self.thermal_head = nn.Linear(hidden_size, T * G)  # logits
        self.storage_head = nn.Linear(
            hidden_size, T * S * 3
        )  # logits, 3 classes (idle, charge, discharge)

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        profiles = x["profiles"].reshape(x["profiles"].shape[0], -1)
        init_conds = x["initial_conditions"].reshape(
            x["initial_conditions"].shape[0], -1
        )
        x = torch.cat([profiles, init_conds], dim=-1)
        h = self.trunk(x)

        # Thermal decisions
        thermal_logits = self.thermal_head(h).view(-1, self.G, self.T)
        thermal_on = torch.sigmoid(thermal_logits)

        # Storage, 3 classes
        storage_logits = self.storage_head(h).view(-1, self.S, self.T, 3)

        # Get hard storage decisions
        y = F.gumbel_softmax(storage_logits, tau=self.tau, hard=True, dim=-1)

        is_charging = y[..., 1]
        is_discharging = y[..., 2]

        return {
            "is_on": thermal_on,
            "thermal_logits": thermal_logits,  # to be used in the loss
            "is_charging": is_charging,
            "is_discharging": is_discharging,
            "storage_logits": storage_logits,  # to be used in the loss
        }
