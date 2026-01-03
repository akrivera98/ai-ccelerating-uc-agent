import torch.nn as nn
import torch


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
    def __init__(self, ed_data_dict, violations_penalty=10.0):
        super().__init__()
        self.profiled_units_cost = None  # TODO: pull later from data dict
        self.start_up_costs = None  # TODO: pull later from data dict (this should be min_power_cost i think)
        self.segment_costs = None  # TODO: pull later from data dict
        self.charge_costs = None  # TODO: pull later from data dict
        self.power_balance_penalty = None  # TODO: pull later from data dict
        self.discharge_costs = None  # TODO: pull later from data dict

        self.min_uptimes = None  # TODO: pull later from data dict
        self.min_downtimes = None  # TODO: pull later from data dict

        self.bce_loss = nn.BCELoss()

    def get_economic_dispatch_cost(self, ed_solution, nn_outputs):
        profiled_generation = ed_solution[0]  # TODO: check the order here
        storage_level = ed_solution[1]
        charge_rate = ed_solution[2]
        discharge_rate = ed_solution[3]
        seg_prod = ed_solution[4]
        prod_above = ed_solution[5]
        curtailment = ed_solution[6]
        commitment = nn_outputs[:, 51]  # check dims later

        # Compute profiled units cost
        profiled_units_cost = torch.sum(self.profiled_units_cost * profiled_generation)

        # Compute storage units costs
        storage_charge_cost = torch.sum(self.charge_costs * charge_rate)
        storage_discharge_cost = torch.sum(self.discharge_costs * discharge_rate)

        # Compute thermal generators costs
        turn_on_costs = torch.sum(
            self.start_up_costs * commitment
        )  # TODO: check switch_on logic later
        segment_production_costs = torch.sum(self.segment_costs * seg_prod)

        # Compute curtailment costs
        curtailment_costs = self.power_balance_penalty * torch.sum(curtailment)

        return (
            profiled_units_cost
            + storage_charge_cost
            + storage_discharge_cost
            + turn_on_costs
            + segment_production_costs
            + curtailment_costs
        )

    def forward(self, ed_solution, nn_outputs, targets, inital_status):
        economic_dispatch_cost = self.get_economic_dispatch_cost(
            ed_solution, nn_outputs
        )

        supervised_loss_term = self.bce_loss(nn_outputs, targets)

        # Evaluate uptime/downtime constraints
        uptime_violations, downtime_violations = _evaluate_uptime_downtime_constraints(
            nn_outputs,
            self.min_uptimes,
            self.min_downtimes,
            initial_status=inital_status,
        )

        uptime_violation_cost = torch.sum(uptime_violations)
        downtime_violation_cost = torch.sum(downtime_violations)

        return (
            economic_dispatch_cost
            + supervised_loss_term
            + self.violations_penalty
            * (uptime_violation_cost + downtime_violation_cost)
        )


def _evaluate_uptime_downtime_constraints(
    outputs: torch.tensor,
    min_uptimes: torch.tensor,
    min_downtimes: torch.tensor,
    initial_status: torch.tensor,
):
    """
    Evaluate minimum uptime violations for thermal generators.

    Args:
        outputs (torch.tensor): Binary tensor indicating on/off status of generators.
        min_uptimes (torch.tensor): Tensor indicating minimum uptime requirements.
    Returns:
        constraint_violations (torch.tensor): Tensor indicating minimum uptime violations.
    """

    outputs = outputs.view(
        72, 51
    )  # (T, num_units) # TODO: check if this screws up the graph

    switch_on, switch_off = _compute_switch_on_off(
        outputs, initial_status=torch.zeros(51)
    )  # (T, num_units)

    uptime_violations = _get_min_uptime_violations(
        switch_on, outputs, min_uptimes, initial_status
    )  # TODO: check the initial status logic
    downtime_violations = _get_min_downtime_violations(
        switch_off, outputs, min_downtimes, initial_status
    )  # TODO: check the initial status logic

    return uptime_violations, downtime_violations  # decide on a norm later


def _compute_switch_on_off(outputs: torch.tensor, initial_status: torch.tensor):
    """
    Compute switch-on events for thermal generators.

    Args:
        outputs (torch.tensor): Binary tensor indicating on/off status of generators. Dimensions: (T, num_units)
        initial_status (torch.tensor): Initial on/off status of generators. Dimensions: (num_units, )
    Returns:
        switch_on_events (torch.tensor): Tensor indicating switch-on events.
    """
    prev = torch.vstack(
        (initial_status.unsqueeze(0), outputs[:-1])
    )  # (T, num_units), exclude last time step

    switch_on = (prev == 0) & (outputs == 1)  # (T, num_units)
    switch_off = (prev == 1) & (outputs == 0)  # (T, num_units)

    return switch_on.double(), switch_off.double()


def _get_min_uptime_violations(
    switch_on: torch.tensor,
    is_on: torch.tensor,
    min_uptimes: torch.tensor,
    initial_status: torch.tensor = None,
):
    """
    Compute minimum uptime violations for thermal generators.

    Args:
        switch_on (torch.tensor): Tensor indicating switch-on events. Dimensions: (T, num_units)
        min_uptimes (torch.tensor): Tensor indicating minimum uptime requirements. Dimensions: (num_units, )
    Returns:
        constraint_violations (torch.tensor): Tensor indicating minimum uptime violations.
    """
    T, num_units = switch_on.shape

    # Get prefix sum
    prefix_sum_switch_on = torch.cumsum(switch_on, dim=0)

    # Get switch on sum
    start_indices = (
        torch.arange(T).unsqueeze(1).repeat(1, num_units) - min_uptimes.unsqueeze(0)
    ).clamp(min=0)
    switch_on_sum = (
        prefix_sum_switch_on
        - prefix_sum_switch_on[start_indices, torch.arange(num_units).unsqueeze(0)]
    )

    # Compute in-horizon violations
    in_violations = torch.maximum(
        torch.zeros_like(switch_on_sum), switch_on_sum - is_on
    )

    # Compute pre-horizon violations
    initial_on_time = torch.maximum(torch.zeros_like(initial_status), initial_status)
    remaining_uptime = torch.maximum(
        torch.zeros_like(min_uptimes), min_uptimes - initial_on_time
    )
    early_mask = torch.arange(T).unsqueeze(1).repeat(
        1, num_units
    ) < remaining_uptime.unsqueeze(0)
    pre_violations = early_mask * torch.clamp(1.0 - is_on, min=0.0)

    violations = in_violations + pre_violations

    return violations


def _get_min_downtime_violations(
    switch_off: torch.tensor,
    is_on: torch.tensor,
    min_downtimes: torch.tensor,
    initial_status: torch.tensor = None,
):
    T, num_units = switch_off.shape

    # Get prefix sum
    prefix_sum_switch_off = torch.cumsum(switch_off, dim=0)

    # Get switch off sum
    start_indices = (
        torch.arange(T).unsqueeze(1).repeat(1, num_units) - min_downtimes.unsqueeze(0)
    ).clamp(min=0)
    switch_off_sum = (
        prefix_sum_switch_off
        - prefix_sum_switch_off[start_indices, torch.arange(num_units).unsqueeze(0)]
    )

    # Compute in-horizon violations
    in_violations = torch.maximum(
        torch.zeros_like(switch_off_sum), switch_off_sum - is_on + 1
    )

    # Compute pre-horizon violations
    initial_off_time = torch.maximum(
        torch.zeros_like(initial_status), initial_status
    )  # TODO: this is wrong, it can't be the same logic as uptime
    remaining_downtime = torch.maximum(
        torch.zeros_like(min_downtimes), min_downtimes - initial_off_time
    )
    early_mask = torch.arange(T).unsqueeze(1).repeat(
        1, num_units
    ) < remaining_downtime.unsqueeze(0)
    pre_violations = early_mask * torch.clamp(1.0 - is_on, min=0.0)

    violations = in_violations + pre_violations

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
