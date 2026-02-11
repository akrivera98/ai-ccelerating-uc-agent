import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from src.ed_models.data_classes import create_data_dict
import torch.nn.functional as F
from src.registry import registry

class BaseLossED(nn.Module):
    """
    Base loss for UC + ED training.

    DEFAULT BEHAVIOR:
      - expects `ed_solution` (primal variables)
      - computes ED cost from the primal (like your current CustomLoss_ED)

    Child classes may override:
      - compute_ed_term(...)      (e.g. SciPy dual-based objective)
      - compute_supervised_term(...)
      - compute_violation_term(...)
    """

    def __init__(self, *, instance_path: str, weights: Dict[str, Optional[float]]):
        super().__init__()

        # ------------------
        # Supervised losses
        # ------------------
        self.is_on_sup_loss = nn.BCELoss()
        self.storage_sup_loss = nn.CrossEntropyLoss()

        # ------------------
        # Weights
        # ------------------
        self.w_ed = float(weights.get("ed_objective", 1.0) or 0.0)
        self.w_sup = float(weights.get("supervised", 1.0) or 0.0)
        self.w_viol = float(weights.get("violations", 1.0) or 0.0)

        # ------------------
        # Load ED / UC params
        # ------------------
        self._load_instance_parameters(instance_path)

        self.loss_names = ["total", "ed", "supervised", "violations"]

    # ======================================================================
    # Forward
    # ======================================================================
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        nn_outputs: Dict[str, torch.Tensor],
        ed_solution: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:

        # ---- initial commitment/status ----
        initial_status = features["initial_conditions"][:, -1, :]  # (B,G)
        initial_commitment = initial_status > 0

        is_on = nn_outputs["is_on"]  # (B,T,G), REQUIRED

        # ---- switch events ----
        switch_on, switch_off = self.compute_switch_events(is_on, initial_commitment)

        # ---- compute raw terms (unweighted) ----
        supervised_raw = (
            self.compute_supervised_term(nn_outputs, targets)
            if self.w_sup > 0
            else self._zeroscalar(is_on)
        )

        violations_raw = (
            self.compute_violation_term(
                is_on=is_on,
                switch_on=switch_on,
                switch_off=switch_off,
                initial_status=initial_status,
            )
            if self.w_viol > 0
            else self._zeroscalar(is_on)
        )

        ed_raw = (
            self.compute_ed_term(ed_solution, nn_outputs)
            if self.w_ed > 0
            else self._zeroscalar(is_on)
        )

        # ---- apply weights ----
        supervised = self.w_sup * supervised_raw
        violations = self.w_viol * violations_raw
        ed_term = self.w_ed * ed_raw

        total = supervised + violations + ed_term

        return {
            "total": total,
            "ed": ed_term,
            "supervised": supervised,
            "violations": violations,
        }

    # ED TERM
    def compute_ed_term(
        self,
        ed_solution: Any,
        nn_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        DEFAULT ED term: compute cost from primal ED solution.

        This matches your current CustomLoss_ED behavior.
        """
        if ed_solution is None:
            raise ValueError("ED term enabled but ed_solution was not provided.")

        return self.compute_economic_dispatch_cost(
            ed_solution=ed_solution,
            nn_outputs=nn_outputs,
        )

    # Supervised term
    def compute_supervised_term(
        self,
        nn_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        loss = self.is_on_sup_loss(
            nn_outputs["thermal_probs"],
            targets["is_on"],
        )

        if nn_outputs.get("storage_logits") is not None:
            logits = nn_outputs["storage_logits"]
            y = targets["storage_status"].argmax(dim=-1).long()
            C = logits.shape[-1]
            loss = loss + self.storage_sup_loss(logits.reshape(-1, C), y.reshape(-1))

        return loss

    # Violation term
    def compute_violation_term(
        self,
        *,
        is_on: torch.Tensor,  # (B,T,G)
        switch_on: torch.Tensor,  # (B,T,G)
        switch_off: torch.Tensor,  # (B,T,G)
        initial_status: torch.Tensor,  # (B,G)
    ) -> torch.Tensor:

        uptime_viol, downtime_viol = self._evaluate_uptime_downtime_constraints(
            is_on,
            switch_on,
            switch_off,
            self.min_uptimes,
            self.min_downtimes,
            initial_status=initial_status,
        )

        return uptime_viol.sum() + downtime_viol.sum()

    # ED cost from primal (copied from your CustomLoss_ED, lightly cleaned)
    def compute_economic_dispatch_cost(
        self,
        *,
        ed_solution,
        nn_outputs,
    ) -> torch.Tensor:

        (
            profiled_generation,
            storage_level,
            charge_rate,
            discharge_rate,
            seg_prod,
            prod_above,
            curtailment,
        ) = ed_solution

        # Expect time-first internally
        profiled_generation = profiled_generation.transpose(1, 2)
        charge_rate = charge_rate.transpose(1, 2)
        discharge_rate = discharge_rate.transpose(1, 2)
        seg_prod = seg_prod.transpose(1, 2)

        # ---- profiled units ----
        profiled_cost = (
            self.profiled_units_cost.view(1, 1, -1) * profiled_generation
        ).sum()

        # ---- storage ----
        storage_cost = (self.charge_costs.view(1, 1, -1) * charge_rate).sum() + (
            self.discharge_costs.view(1, 1, -1) * discharge_rate
        ).sum()

        # ---- thermal ----
        startup_cost = (
            self.start_up_costs.view(1, 1, -1)
            * self.compute_switch_events(
                nn_outputs["is_on"],
                nn_outputs["is_on"][:, 0, :] * 0.0,  # dummy; not used
            )[0]
        ).sum()

        segment_cost = (self.segment_cost.squeeze(1) * seg_prod).sum()

        # ---- curtailment ----
        curtailment_cost = self.power_balance_penalty * curtailment.sum()

        return (
            profiled_cost
            + storage_cost
            + startup_cost
            + segment_cost
            + curtailment_cost
        )

    # Switch events
    @staticmethod
    def compute_switch_events(is_on, initial_commitment):
        prev = torch.cat((initial_commitment.unsqueeze(1), is_on[:, :-1, :]), dim=1)
        switch_on = (1.0 - prev) * is_on
        switch_off = prev * (1.0 - is_on)
        return switch_on, switch_off

    def _load_instance_parameters(self, instance_path: str):

        ed_data_dict = create_data_dict(instance_path)

        self.start_up_costs = torch.tensor(
            [g.startup_costs[0] for g in ed_data_dict["thermal_gen_data_list"]],
            dtype=torch.float32,
        )

        self.min_uptimes = torch.tensor(
            [g.min_up_time for g in ed_data_dict["thermal_gen_data_list"]],
            dtype=torch.int64,
        )

        self.min_downtimes = torch.tensor(
            [g.min_down_time for g in ed_data_dict["thermal_gen_data_list"]],
            dtype=torch.int64,
        )

        self.profiled_units_cost = torch.tensor(
            [g.cost for g in ed_data_dict["profiled_gen_data_list"]],
            dtype=torch.float32,
        )

        self.charge_costs = torch.tensor(
            [s.charge_cost for s in ed_data_dict["storage_data_list"]],
            dtype=torch.float32,
        )

        self.discharge_costs = torch.tensor(
            [s.discharge_cost for s in ed_data_dict["storage_data_list"]],
            dtype=torch.float32,
        )

        self.power_balance_penalty = ed_data_dict["system_data"].power_balance_penalty

        # segment cost
        curves = [
            g.production_cost_curve for g in ed_data_dict["thermal_gen_data_list"]
        ]
        max_k = max(len(c) for c in curves)
        _, self.segment_cost = self._build_segment_mw_cost(curves, max_k)

    @staticmethod
    def _build_segment_mw_cost(curves, max_segments):
        G = len(curves)
        segment_mw = torch.zeros((G, 1, max_segments))
        segment_cost = torch.zeros((G, 1, max_segments))
        for i, segs in enumerate(curves):
            for k, (mw, cost) in enumerate(segs):
                segment_mw[i, 0, k] = mw
                segment_cost[i, 0, k] = cost
        return segment_mw, segment_cost

    @staticmethod
    def _zeroscalar(x):
        return torch.zeros((), device=x.device, dtype=x.dtype)

    def _evaluate_uptime_downtime_constraints(
        self,
        is_on: torch.tensor,
        switch_on: torch.tensor,
        switch_off: torch.tensor,
        min_uptimes: torch.tensor,
        min_downtimes: torch.tensor,
        initial_status: torch.tensor,
    ):
        """
        Evaluate minimum uptime violations for thermal generators.

        Args:
            is_on (torch.tensor): Binary tensor indicating on/off status of generators.
            min_uptimes (torch.tensor): Tensor indicating minimum uptime requirements.
        Returns:
            constraint_violations (torch.tensor): Tensor indicating minimum uptime violations.
        """

        uptime_violations = self._get_min_uptime_violations(
            switch_on, switch_off, is_on, min_uptimes, initial_status
        )  # TODO: check the initial status logic
        downtime_violations = self._get_min_downtime_violations(
            switch_on, switch_off, is_on, min_downtimes, initial_status
        )  # TODO: check the initial status logic

        return uptime_violations, downtime_violations  # decide on a norm later

    def _get_min_uptime_violations(  # TODO: CHECK THE LOGIC HERE
        self,
        switch_on: torch.Tensor,  # (B, T, G)
        switch_off: torch.Tensor,  # (B, T, G)
        is_on: torch.Tensor,  # (B, T, G)
        min_uptimes: torch.Tensor,  # (G,)
        initial_status: torch.Tensor,  # (B, G)
    ) -> torch.Tensor:
        B, T, G = switch_on.shape
        device = is_on.device
        dtype = is_on.dtype

        U = min_uptimes.to(device=device).long().clamp(min=0)  # (G,)

        violations = torch.zeros((B, T, G), device=device, dtype=dtype)

        # Minimum uptime for initial periods
        t_idx = torch.arange(T, device=device).view(1, T, 1)  # (1,T,1)
        init_on_time = initial_status.to(device=device).clamp(min=0).long()  # (B,G)
        remaining_uptime = torch.clamp(
            U.unsqueeze(0) - init_on_time, min=0
        ).long()  # (B,G)

        has_to_be_on_mask = (t_idx < remaining_uptime.unsqueeze(1)) & (
            initial_status.unsqueeze(1) > 0
        )  # (B, T, G)

        violations = violations + (switch_off * has_to_be_on_mask).sum(
            dim=-1
        ).unsqueeze(-1)  # (B, T, G)

        # Minimum uptime for the rest of the periods

        # Prefix sum on switch_on
        cumsum_switch_on = torch.cumsum(switch_on, dim=1)  # (B, T, G)

        V = torch.cat(
            [torch.zeros((B, T, 1), device=device, dtype=dtype), cumsum_switch_on],
            dim=-1,
        )  # (B, T, G+1) adds a leading 0 so prefix sum at -1 is 0
        U_exp = U.view(1, 1, G)  # (1, 1, G)
        idx = (t_idx - U_exp + 1).clamp(
            min=0, max=T
        )  # index where the rolling window starts, + 1 because of the leading 0 in V

        V_shifted = V.gather(dim=1, index=idx.expand(B, T, G))

        window_sum = cumsum_switch_on - V_shifted  # (B, T, G)

        violations = F.relu(window_sum - is_on)  # (B, T, G)
        return violations

    def _get_min_downtime_violations(  # TODO: CHECK THE LOGIC HERE
        self,
        switch_on: torch.Tensor,  # (B, G, T)
        switch_off: torch.Tensor,  # (B, G, T)
        is_on: torch.Tensor,  # (B, G, T)
        min_downtimes: torch.Tensor,  # (G,)
        initial_status: torch.Tensor,  # (B, G)
    ) -> torch.Tensor:
        B, T, G = switch_on.shape
        device = is_on.device
        dtype = is_on.dtype

        D = min_downtimes.to(device=device).long().clamp(min=0)  # (G,)

        violations = torch.zeros((B, T, G), device=device, dtype=dtype)

        # Minimum uptime for initial periods
        t_idx = torch.arange(T, device=device).view(1, T, 1)  # (1, T, 1)
        init_off_time = initial_status.to(device=device).clamp(max=0).long()  # (B,G)
        remaining_downtime = torch.clamp(
            D.unsqueeze(0) - init_off_time, min=0
        ).long()  # (B,G)

        has_to_be_off_mask = (t_idx < remaining_downtime.unsqueeze(1)) & (
            initial_status.unsqueeze(1) == 0
        )  # (B, T, G)

        violations = violations + (switch_on * has_to_be_off_mask).sum(
            dim=-1
        ).unsqueeze(-1)  # (B, T, G)

        # Minimum uptime for the rest of the periods

        # Prefix sum on switch_off
        cumsum_switch_off = torch.cumsum(switch_off, dim=1)  # (B, T, G)

        V = torch.cat(
            [torch.zeros((B, T, 1), device=device, dtype=dtype), cumsum_switch_off],
            dim=-1,
        )  # (B, T, G+1) adds a leading 0 so prefix sum at -1 is 0
        D_exp = D.view(1, 1, G)  # (1, 1, G)
        idx = (t_idx - D_exp + 1).clamp(
            min=0, max=T
        )  # index where the rolling window starts, + 1 because of the leading 0 in V

        V_shifted = V.gather(dim=1, index=idx.expand(B, T, G))

        window_sum = cumsum_switch_off - V_shifted  # (B, T, G)

        violations = F.relu(window_sum - (1 - is_on))  # (B, T, G)

        return violations


@registry.register_loss("cvxpylayers_loss")
class CvxpyLayersLoss(BaseLossED):
    pass

@registry.register_loss("scipy_solver_loss")
class ScipySolverLoss(BaseLossED):
    """
    SciPy/HiGHS version: `ed_solution` is the differentiable ED objective tensor
    produced by your custom autograd Function (dual-based backward).

    Trainer should pass:
        ed_solution = ed_obj
    where ed_obj is:
        - scalar tensor, or
        - (B,) tensor (one objective per batch item)
    """

    def compute_ed_term(
        self,
        ed_solution: torch.Tensor,
        nn_outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if ed_solution is None:
            raise ValueError(
                "ScipySolverLoss: ED term enabled but ed_solution was None. "
                "Pass the objective tensor from ed_model.objective(...)."
            )
        if not torch.is_tensor(ed_solution):
            raise TypeError(
                f"ScipySolverLoss: expected ed_solution to be a torch.Tensor, got {type(ed_solution)}"
            )

        # Reduce to scalar for the total loss
        if ed_solution.dim() == 0:
            return ed_solution
        if ed_solution.dim() == 1:
            return ed_solution.mean()

        raise ValueError(
            f"ScipySolverLoss: expected objective tensor to be scalar or (B,), got {tuple(ed_solution.shape)}"
        )

@registry.register_loss("qpth_solver_loss")
class QpthSolverLoss(BaseLossED):
    pass # Need to implement here custom methods