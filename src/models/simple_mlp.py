from torch import nn
from src.models.registry import register_model

@register_model("Simple_MLP")
def build_simple_mlp_model(cfg):
    return SimpleMLP(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        output_size=cfg.output_size,
        num_hidden_layers=cfg.num_hidden_layers,
        final_activation=cfg.final_activation,
    )

class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_hidden_layers: int,
        final_activation: str = "",
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