from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional, Sequence

from src.models.round import ste_round
from src.registry import registry


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

@registry.register_model("flexible_two_head_mlp")
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
