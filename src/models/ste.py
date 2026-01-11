"""
Straight-through estimators for nondifferentiable operators
"""

import torch
from torch import nn
from torch.autograd import Function

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

