import torch
import torch.nn as nn
from qpth.qp import QPFunction, QPSolvers

from src.ed_models.cannon import EDFormulation, EDRHSBuilder
from src.registry import registry

# TODO: this hasn't been tested

class EDQPTHSolver(nn.Module):
    """
    qpth solver adapter.

    Takes:
      - form: EDFormulation (compiled A,G,Q_base,q_base + metadata)
      - rhs:  EDRHSBuilder (builds batched q,h,b)

    Solves (batched):
        min_x  0.5 x^T Q x + q^T x
        s.t.   G x <= h
               A x  = b

    """

    def __init__(self, form, rhs_builder: nn.Module, *, solver=QPSolvers.PDIPM_BATCHED):
        super().__init__()
        self.form = form
        self.rhs = rhs_builder
        self.solver = solver

        # qpth expects Q as (nz,nz) and q,G,h,A,b as batched
        # We'll use the formulation's stored Q_base by default.
        # (If you want to override eps at runtime, see solve(..., Q_override=...))
        self.register_buffer("_Q", form.Q_base)

    @torch.no_grad()
    def sanity_check_shapes(self):
        # Quick check that matrices exist and sizes match
        assert self.form.A.shape == (self.form.neq, self.form.nz)
        assert self.form.G.shape == (self.form.nineq, self.form.nz)
        assert self._Q.shape == (self.form.nz, self.form.nz)
        assert self.form.q_base.shape == (self.form.nz,)

    def solve(
        self,
        load,
        solar_max,
        wind_max,
        is_on,
        is_charging,
        is_discharging,
        *,
        verbose=False,
        Q_override=None,
        q_override=None,
    ):
        """
        Returns:
          x: (B, nz) primal solution from qpth

        Optional overrides:
          - Q_override: (nz,nz) tensor to use instead of form.Q_base
          - q_override: (B,nz) or (nz,) tensor to use instead of form.q_base
                        (useful if you later make q depend on inputs)
        """
        q, h, b, *_ = self.rhs.build(
            load, solar_max, wind_max, is_on, is_charging, is_discharging
        )

        # Allow overriding q (objective vector)
        if q_override is not None:
            if q_override.dim() == 1:
                q = (
                    q_override.to(device=q.device, dtype=q.dtype)
                    .unsqueeze(0)
                    .expand_as(q)
                )
            else:
                q = q_override.to(device=q.device, dtype=q.dtype)

        # Allow overriding Q
        Q = (
            self._Q
            if Q_override is None
            else Q_override.to(device=q.device, dtype=q.dtype)
        )

        # qpth call (batched)
        x = QPFunction(verbose=verbose, solver=self.solver)(
            Q,  # (nz,nz)
            q,  # (B,nz)
            self.form.G,  # (nineq,nz)
            h,  # (B,nineq)
            self.form.A,  # (neq,nz)
            b,  # (B,neq)
        )
        return x


@registry.register_ed_model("qpth")
class EDModelQP(nn.Module):
    """
    Convenience wrapper that looks like your old EDModelQP, but is cleanly factored:
      - form: EDFormulation
      - rhs:  EDRHSBuilder
      - solver: EDQPTHSolver
    """

    def __init__(
        self,
        ed_data_dict,
        *,
        eps=1e-3,
        device="cpu",
        dtype=torch.float32,
        solver=QPSolvers.PDIPM_BATCHED,
    ):
        super().__init__()
        self.form = EDFormulation(ed_data_dict, eps=eps, device=device, dtype=dtype)
        self.rhs = EDRHSBuilder(self.form)
        self.solver = EDQPTHSolver(self.form, self.rhs, solver=solver)

    def forward(
        self,
        load,
        solar_max,
        wind_max,
        is_on,
        is_charging,
        is_discharging,
        *,
        verbose=False,
    ):
        return self.solver.solve(
            load,
            solar_max,
            wind_max,
            is_on,
            is_charging,
            is_discharging,
            verbose=verbose,
        )
