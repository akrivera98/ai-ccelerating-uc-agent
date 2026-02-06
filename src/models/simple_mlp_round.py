from torch import nn
from src.models.registry import register_model
from src.models.round import RoundModel



@register_model("MLP_with_rounding")
def build_mlp_with_rounding_model(cfg):
    return MLP_with_rounding(
        input_size=cfg.input_size,
        hidden_size=cfg.hidden_size,
        output_size=cfg.output_size,
        num_hidden_layers=cfg.num_hidden_layers,
        rounding_strategy=cfg.rounding_strategy,
    )


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
    