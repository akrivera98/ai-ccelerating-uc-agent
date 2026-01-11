
import torch.nn as nn
import torch
from src.models.ste import diffFloor, diffGumbelBinarize, thresholdBinarize

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