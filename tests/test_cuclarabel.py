import juliacall  # keep first
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

device = torch.device("cuda")

n = 50
x = cp.Variable(n)
A = cp.Parameter((n, n))
b = cp.Parameter(n)

constraints = [x >= 0]
objective = cp.Minimize(cp.norm(A @ x - b, 2))   # SOC form (no quadratic P)
prob = cp.Problem(objective, constraints)

layer = CvxpyLayer(prob, parameters=[A, b], variables=[x],
                   solver=cp.CUCLARABEL).to(device)

A_t = torch.randn(n, n, device=device, requires_grad=True)
b_t = torch.randn(n, device=device, requires_grad=True)

(x_sol,) = layer(A_t, b_t)
loss = x_sol.sum()
loss.backward()

print("ok:", x_sol.is_cuda, A_t.grad is not None, b_t.grad is not None)
