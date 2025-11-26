import torch

class StraightThroughRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through = pass the gradient as if no rounding happened
        return grad_output.clone()


def ste_round(x):
    return StraightThroughRound.apply(x)
