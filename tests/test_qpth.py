import torch
import torch.nn.functional as F

from qpth.qp import QPFunction, QPSolvers
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable

class OptNet(nn.Module):
    def __init__(self, nFeatures, nHidden, nCls, bn, nineq=200, neq=0, eps=1e-4):
        super().__init__()

        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(nCls)

        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)

        self.M = Variable(torch.tril(torch.ones(nCls, nCls)))
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls)))
        self.G = Parameter(torch.Tensor(nineq,nCls).uniform_(-1,1))
        self.z0 = Parameter(torch.zeros(nCls))
        self.s0 = Parameter(torch.ones(nineq))

    def forward(self, x):
        nBatch = x.size(0)

        # FC-ReLU-(BN)-FC-ReLU-(BN)-QP-Softmax
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)

        L = self.M*self.L
        Q = L.mm(L.t()) + self.eps*Variable(torch.eye(self.nCls))
        h = self.G.mv(self.z0)+self.s0
        e = Variable(torch.Tensor())
        x = QPFunction(verbose=True, solver=QPSolvers.CVXPY)(Q, x, self.G, h, e, e)

        return F.log_softmax(x)


def main():
    torch.manual_seed(0)

    device = torch.device("cpu")

    # Small dimensions for fast solve
    nFeatures = 10
    nHidden = 16
    nCls = 7
    nineq = 20
    neq = 0
    B = 4

    model = OptNet(
        nFeatures=nFeatures,
        nHidden=nHidden,
        nCls=nCls,
        bn=False,
        nineq=nineq,
        neq=neq,
        eps=1e-4,
    ).to(device)

    x = torch.randn(B, nFeatures, device=device, requires_grad=True)

    print("Running forward pass...")
    out = model(x)
    print("Output shape:", out.shape)
    assert out.shape == (B, nCls)
    assert torch.isfinite(out).all()

    print("Running backward pass...")
    y = torch.randint(0, nCls, (B,), device=device)
    loss = F.nll_loss(out, y)
    loss.backward()

    print("Loss:", loss.item())

    # Gradient sanity checks
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"⚠️  No grad for {name}")
        else:
            assert torch.isfinite(p.grad).all(), f"Non-finite grad in {name}"

    print("✅ OptNet QP forward/backward test passed.")


if __name__ == "__main__":
    main()
