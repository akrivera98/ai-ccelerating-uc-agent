import torch.nn as nn
import torch
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.mse_loss(inputs, targets)


class CustomLoss1(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.lp_infeasibility_penalty = 1.0

    def compute_lp_infeasibility(self, targets):
        # Placeholder for LP infeasibility computation
        return torch.tensor(0.0)  # Replace with actual computation

    def compute_integer_violations(self, targets, params):
        return None

    def forward(self, inputs, targets, params):
        bce_cost = self.bce_loss(inputs, targets)

        return None


class CustomLoss(nn.Module):
    def __init__(self, ed_data_dict, loss_weights, solve_lp_in_loss=True):
        super().__init__()

        self._get_parameters_from_data_dict(ed_data_dict)

        self.is_on_sup_loss = nn.BCELoss()
        self.is_charging_sup_loss = nn.BCELoss()
        self.is_discharging_sup_loss = nn.BCELoss()
        self.violations_penalty = (
            loss_weights.violation if loss_weights.violation is not None else 1.0
        )
        self.supervised_weight = (
            loss_weights.supervised if loss_weights.supervised is not None else 1.0
        )
        self.ed_objective_weight = (
            loss_weights.ed_objective if loss_weights.ed_objective is not None else 1.0
        )
        self.start_up_weight = (
            loss_weights.startup if loss_weights.startup is not None else 1.0
        )

        self.solve_lp_in_loss = solve_lp_in_loss

    def _get_parameters_from_data_dict(self, ed_data_dict):
        production_cost_curves_list = [
            torch.tensor(g.production_cost_curve)
            for g in ed_data_dict["thermal_gen_data_list"]
        ]

        max_segments = max(
            len(g.production_cost_curve) for g in ed_data_dict["thermal_gen_data_list"]
        )

        _, self.segment_cost = self._build_segment_mw_cost(
            production_cost_curves_list, max_segments=max_segments
        )

        startup_costs_list = [
            g.startup_costs[0] for g in ed_data_dict["thermal_gen_data_list"]
        ]
        self.start_up_costs = torch.tensor(
            startup_costs_list, dtype=torch.float32
        )  # TODO: add dims

        min_uptimes_list = [
            g.min_up_time for g in ed_data_dict["thermal_gen_data_list"]
        ]
        self.min_uptimes = torch.tensor(
            min_uptimes_list, dtype=torch.int64
        )  # TODO: add dims

        min_downtimes_list = [
            g.min_down_time for g in ed_data_dict["thermal_gen_data_list"]
        ]
        self.min_downtimes = torch.tensor(
            min_downtimes_list, dtype=torch.int64
        )  # TODO: add dims

        profiled_units_cost_list = [
            g.cost for g in ed_data_dict["profiled_gen_data_list"]
        ]
        self.profiled_units_cost = torch.tensor(
            profiled_units_cost_list, dtype=torch.float32
        )  # TODO: add dims

        storage_charge_costs_list = [
            s.charge_cost for s in ed_data_dict["storage_data_list"]
        ]
        self.charge_costs = torch.tensor(
            storage_charge_costs_list, dtype=torch.float32
        )  # TODO: add dims

        storage_discharge_costs_list = [
            s.discharge_cost for s in ed_data_dict["storage_data_list"]
        ]
        self.discharge_costs = torch.tensor(
            storage_discharge_costs_list, dtype=torch.float32
        )  # TODO: add dims

        self.power_balance_penalty = ed_data_dict["system_data"].power_balance_penalty

    def get_economic_dispatch_cost(self, ed_solution, nn_outputs, switch_on):
        profiled_generation = ed_solution[0]  # TODO: check the order here
        storage_level = ed_solution[1]
        charge_rate = ed_solution[2]
        discharge_rate = ed_solution[3]
        seg_prod = ed_solution[4]
        prod_above = ed_solution[5]
        curtailment = ed_solution[6]
        commitment = nn_outputs["is_on"]

        # Compute profiled units cost
        profiled_units_cost = (
            self.profiled_units_cost.view(1, -1, 1) * profiled_generation
        ).sum()

        # Compute storage units costs
        storage_charge_cost = (self.charge_costs.view(1, -1, 1) * charge_rate).sum()
        storage_discharge_cost = (
            self.discharge_costs.view(1, -1, 1) * discharge_rate
        ).sum()

        # Compute curtailment costs
        curtailment_costs = self.power_balance_penalty * torch.sum(curtailment)

        # Compute thermal generators costs
        turn_on_costs = (self.start_up_costs.view(1, -1, 1) * switch_on).sum()

        segment_production_costs = (self.segment_cost.unsqueeze(0) * seg_prod).sum()

        return (
            profiled_units_cost
            + storage_charge_cost
            + storage_discharge_cost
            + turn_on_costs
            + segment_production_costs
            + curtailment_costs
        )

    def _build_segment_mw_cost(self, curves, max_segments):
        num_units = len(curves)
        segment_mw = torch.zeros((num_units, 1, max_segments))
        segment_cost = torch.zeros((num_units, 1, max_segments))

        for i, seg_list in enumerate(curves):
            all_mw = [mw for mw, cost in seg_list]
            all_cost = [cost for mw, cost in seg_list]
            n = len(all_mw)
            segment_mw[i, 0, :n] = torch.tensor(all_mw)
            segment_cost[i, 0, :n] = torch.tensor(all_cost)

        return segment_mw, segment_cost

    def compute_startup_costs(self, switch_on):
        return (self.start_up_costs.view(1, -1, 1) * switch_on).sum()

    def forward(
        self,
        ed_model_lp,
        outputs_dict,
        targets,
        initial_status,
        initial_commitment,
        load,
        solar_max,
        wind_max,
    ):
        # Supervised loss terms
        supervised_loss_term = (
            self.compute_supervised_loss(outputs_dict, targets) * self.supervised_weight
        )

        self.switch_on, self.switch_off = self._compute_switch_on_off(
            outputs_dict["is_on_rounded"], initial_commitment=initial_commitment
        )

        # # Unsupervised loss terms
        # economic_dispatch_cost = self.get_economic_dispatch_cost(
        #     ed_solution, outputs_dict, self.switch_on
        # )  # TODO: check that this cost and the LP objective are the same.

        # LP objective (I'm getting grads wrt to this)
        if self.solve_lp_in_loss:
            is_on = outputs_dict["is_on_rounded"]
            is_charging = outputs_dict["is_charging"]
            is_discharging = outputs_dict["is_discharging"]
            economic_dispatch_cost = (
                ed_model_lp.objective(
                    load, solar_max, wind_max, is_on, is_charging, is_discharging
                ).mean()
                * self.ed_objective_weight
            ) / 10000.0 # trying to scale down the ED objective
        else:
            economic_dispatch_cost = torch.tensor(0.0, device=load.device)

        # turn on costs
        startup_costs = (
            self.compute_startup_costs(self.switch_on) * self.start_up_weight
        ) / load.shape[0] # normalize by batch size

        # Constraint violation loss terms
        up_down_time_violations_cost = (
            self.compute_constraint_violations(
                outputs_dict["is_on_rounded"],
                self.switch_on,
                self.switch_off,
                initial_status=initial_commitment,
            )
            * self.violations_penalty
        ) / load.shape[0]  # normalize by batch size
        total = (
            economic_dispatch_cost
            + startup_costs
            + supervised_loss_term
            + up_down_time_violations_cost
        )

        return {
            "total": total,
            "ed": economic_dispatch_cost,
            "startup": startup_costs,
            "supervised": supervised_loss_term,
            "violations": up_down_time_violations_cost,
        }

    def compute_supervised_loss(self, outputs_dict, targets):
        thermal_commitment_loss = self.is_on_sup_loss(
            outputs_dict["is_on"], targets["is_on"]
        )

        is_charging_loss = self.is_charging_sup_loss(
            outputs_dict["is_charging"], targets["is_charging"]
        )

        is_discharging_loss = self.is_discharging_sup_loss(
            outputs_dict["is_discharging"], targets["is_discharging"]
        )

        supervised_loss_term = (
            thermal_commitment_loss + is_charging_loss + is_discharging_loss
        )

        return supervised_loss_term

    def compute_constraint_violations(
        self, is_on, switch_on, switch_off, initial_status
    ):
        uptime_violations, downtime_violations = _evaluate_uptime_downtime_constraints(
            is_on,
            switch_on,
            switch_off,
            self.min_uptimes,
            self.min_downtimes,
            initial_status=initial_status,
        )

        violations_cost = uptime_violations.sum() + downtime_violations.sum()

        return self.violations_penalty * violations_cost

    def _compute_switch_on_off(
        self, outputs: torch.tensor, initial_commitment: torch.tensor
    ):
        """
        Compute switch-on events for thermal generators.

        Args:
            outputs (torch.tensor): Binary tensor indicating on/off status of generators. Dimensions: (T, num_units)
            initial_commitment (torch.tensor): Initial on/off status of generators. Dimensions: (num_units, )
        Returns:
            switch_on_events (torch.tensor): Tensor indicating switch-on events.
        """
        prev = torch.cat(
            (initial_commitment.unsqueeze(-1), outputs[:, :, :-1]), dim=2
        )  # (G, T), excluding last time step

        switch_on = (1.0 - prev) * outputs
        switch_off = prev * (1.0 - outputs)  # (B, G, T)

        return switch_on, switch_off


def _evaluate_uptime_downtime_constraints(
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

    uptime_violations = _get_min_uptime_violations(
        switch_on, switch_off, is_on, min_uptimes, initial_status
    )  # TODO: check the initial status logic
    downtime_violations = _get_min_downtime_violations(
        switch_on, switch_off, is_on, min_downtimes, initial_status
    )  # TODO: check the initial status logic

    return uptime_violations, downtime_violations  # decide on a norm later


def _get_min_uptime_violations(  # TODO: CHECK THE LOGIC HERE
    switch_on: torch.Tensor,  # (B, G, T)
    switch_off: torch.Tensor,  # (B, G, T)
    is_on: torch.Tensor,  # (B, G, T)
    min_uptimes: torch.Tensor,  # (G,)
    initial_status: torch.Tensor,  # (B, G)
) -> torch.Tensor:
    B, G, T = switch_on.shape
    device = is_on.device
    dtype = is_on.dtype

    U = min_uptimes.to(device=device).long().clamp(min=0)  # (G,)

    violations = torch.zeros((B, G, T), device=device, dtype=dtype)

    # Minimum uptime for initial periods
    t_idx = torch.arange(T, device=device).view(1, 1, T)  # (1,1,T)
    init_on_time = initial_status.to(device=device).clamp(min=0).long()  # (B,G)
    remaining_uptime = torch.clamp(U.unsqueeze(0) - init_on_time, min=0).long()  # (B,G)

    has_to_be_on_mask = (t_idx < remaining_uptime.unsqueeze(-1)) & (
        initial_status.unsqueeze(-1) > 0
    )  # (B,G,T)

    violations = violations + (switch_off * has_to_be_on_mask).sum(dim=-1).unsqueeze(
        -1
    )  # (B, G)

    # Minimum uptime for the rest of the periods

    # Prefix sum on switch_on
    cumsum_switch_on = torch.cumsum(switch_on, dim=-1)  # (B, G, T)

    V = torch.cat(
        [torch.zeros((B, G, 1), device=device, dtype=dtype), cumsum_switch_on], dim=-1
    )  # (B, G, T+1) adds a leading 0 so prefix sum at -1 is 0
    U_exp = U.view(1, G, 1)  # (1, G, 1)
    idx = (t_idx - U_exp + 1).clamp(
        min=0, max=T
    )  # index where the rolling window starts, + 1 because of the leading 0 in V

    V_shifted = V.gather(dim=-1, index=idx.expand(B, G, T))

    window_sum = cumsum_switch_on - V_shifted  # (B, G, T)

    violations = F.relu(window_sum - is_on)  # (B, G, T)

    return violations


def _get_min_downtime_violations(  # TODO: CHECK THE LOGIC HERE
    switch_on: torch.Tensor,  # (B, G, T)
    switch_off: torch.Tensor,  # (B, G, T)
    is_on: torch.Tensor,  # (B, G, T)
    min_downtimes: torch.Tensor,  # (G,)
    initial_status: torch.Tensor,  # (B, G)
) -> torch.Tensor:
    B, G, T = switch_on.shape
    device = is_on.device
    dtype = is_on.dtype

    D = min_downtimes.to(device=device).long().clamp(min=0)  # (G,)

    violations = torch.zeros((B, G, T), device=device, dtype=dtype)

    # Minimum uptime for initial periods
    t_idx = torch.arange(T, device=device).view(1, 1, T)  # (1,1,T)
    init_off_time = initial_status.to(device=device).clamp(max=0).long()  # (B,G)
    remaining_downtime = torch.clamp(
        D.unsqueeze(0) - init_off_time, min=0
    ).long()  # (B,G)

    has_to_be_off_mask = (t_idx < remaining_downtime.unsqueeze(-1)) & (
        initial_status.unsqueeze(-1) == 0
    )  # (B,G,T)

    violations = violations + (switch_on * has_to_be_off_mask).sum(dim=-1).unsqueeze(
        -1
    )  # (B, G)

    # Minimum uptime for the rest of the periods

    # Prefix sum on switch_off
    cumsum_switch_off = torch.cumsum(switch_off, dim=-1)  # (B, G, T)

    V = torch.cat(
        [torch.zeros((B, G, 1), device=device, dtype=dtype), cumsum_switch_off], dim=-1
    )  # (B, G, T+1) adds a leading 0 so prefix sum at -1 is 0
    D_exp = D.view(1, G, 1)  # (1, G, 1)
    idx = (t_idx - D_exp + 1).clamp(
        min=0, max=T
    )  # index where the rolling window starts, + 1 because of the leading 0 in V

    V_shifted = V.gather(dim=-1, index=idx.expand(B, G, T))

    window_sum = cumsum_switch_off - V_shifted  # (B, G, T)

    violations = F.relu(window_sum - (1 - is_on))  # (B, G, T)

    return violations


def compute_startup_costs(switch_on: torch.tensor, startup_costs: torch.tensor):
    """
    Compute startup costs for thermal generators.
    Args:
        switch_on (torch.tensor): Tensor indicating switch-on events. Dimensions: (T, num_units)
        startup_costs (torch.tensor): Tensor indicating startup costs. Dimensions: (num_units, )
    """
    startup_costs = switch_on * startup_costs.unsqueeze(0)  # (T, num_units)
    total_startup_costs = torch.sum(startup_costs)
    return total_startup_costs


def compute_UC_objective(lp_objective: torch.tensor, startup_costs: torch.tensor):
    """
    This would be the summation of:
    - statup costs
    - the LP objective (fuel and curtailment costs), pulled from the CVXPYlayer ?
    """
    return lp_objective + startup_costs
