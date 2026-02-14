from abc import ABC, abstractmethod
import cvxpy as cp
import numpy as np
from src.ed_models.data_utils import (
    SystemData,
    StorageUnitData,
    ThermalGeneratorData,
)

T = 72  # Number of time periods


class UCComponent(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def define_variables(self):
        """Return a dict of cvxpy Variables."""

    @abstractmethod
    def define_constraints(self):
        """Return a list of cvxpy constraints."""

    @abstractmethod
    def define_objective(self):
        """Return a cvxpy expression or None."""


class ProfiledGeneratorsComponent(UCComponent):
    def __init__(self, model, list_profiled_gen_data):
        super().__init__(model)
        self.data = list_profiled_gen_data
        self.n_gens = len(self.data)
        self.T = model.T

        self.names = [g.name for g in self.data]
        self.cost = np.array([g.cost for g in self.data], dtype=float)  # (n_gens,)
        self.min_power = np.array(
            [g.min_power for g in self.data], dtype=float
        )  # (n_gens,)
        self.max_power = np.array(
            [g.max_power for g in self.data], dtype=float
        )  # (n_gens,)

        # Find indices
        self.idx_solar = self.names.index("solar")
        self.idx_wind = self.names.index("wind")
        self.idx_hydro = [
            i for i, n in enumerate(self.names) if n not in ("solar", "wind")
        ]

    def define_variables(self):
        self.profiled_generation = cp.Variable((self.n_gens, self.T), nonneg=True)
        return {"profiled_generation": self.profiled_generation}

    def define_parameters(self):
        self.max_power_solar = cp.Parameter(self.T, nonneg=True, name="max_power_solar")
        self.max_power_wind = cp.Parameter(self.T, nonneg=True, name="max_power_wind")
        return {
            "max_power_solar": self.max_power_solar,
            "max_power_wind": self.max_power_wind,
        }

    def define_constraints(self):
        constraints = []

        # Lower bounds: (n_gens, 1) so it broadcasts across T
        min_powers = self.min_power[:, None]  # (n_gens, 1)
        constraints.append(self.profiled_generation >= min_powers)

        # Build max_powers_full row-by-row in the SAME order as self.data
        rows = []
        for i, name in enumerate(self.names):
            if name == "solar":
                rows.append(np.reshape(self.max_power_solar, (1, self.T)))  # (1, T)
            elif name == "wind":
                rows.append(np.reshape(self.max_power_wind, (1, self.T)))  # (1, T)
            else:
                rows.append(
                    np.full((1, self.T), self.max_power[i], dtype=float)
                )  # constant (1, T)

        max_powers_full = cp.vstack(rows)  # (n_gens, T)
        constraints.append(self.profiled_generation <= max_powers_full)

        return constraints

    def define_objective(self):
        cost = self.cost[:, None]  # (n_gens, 1)
        return cp.sum(cp.multiply(cost, self.profiled_generation))


class StorageUnitsComponent(UCComponent):
    def __init__(self, model, storage_data_list: list[StorageUnitData]):
        super().__init__(model)
        self.num_units = len(storage_data_list)
        self.max_levels = [data.max_level for data in storage_data_list]
        self.min_levels = [
            data.min_level if hasattr(data, "min_level") else 0.0
            for data in storage_data_list
        ]
        self.charge_costs = [data.charge_cost for data in storage_data_list]
        self.discharge_costs = [data.discharge_cost for data in storage_data_list]
        self.min_charge_rate = [
            data.min_charge_rate if hasattr(data, "min_charge_rate") else 0.0
            for data in storage_data_list
        ]
        self.max_charge_rate = [data.max_charge_rate for data in storage_data_list]
        self.min_discharge_rate = [
            data.min_discharge_rate if hasattr(data, "min_discharge_rate") else 0.0
            for data in storage_data_list
        ]
        self.max_discharge_rate = [
            data.max_discharge_rate for data in storage_data_list
        ]
        self.initial_levels = [data.initial_level for data in storage_data_list]
        self.loss_factors = [
            data.loss_factor if hasattr(data, "loss_factor") else 0.0
            for data in storage_data_list
        ]
        self.charge_efficiencies = [
            data.charge_efficiency for data in storage_data_list
        ]
        self.discharge_efficiencies = [
            data.discharge_efficiency for data in storage_data_list
        ]
        self.min_ending_levels = [
            data.min_ending_level if hasattr(data, "min_ending_level") else 0.0
            for data in storage_data_list
        ]  # these default to min level
        self.max_ending_levels = [
            data.max_ending_level
            if hasattr(data, "max_ending_level")
            else data.max_level
            for data in storage_data_list
        ]  # these default to max level

    def define_parameters(self):
        self.is_charging = cp.Parameter(
            (self.num_units, self.model.T)
        )  # check this syntax
        self.is_discharging = cp.Parameter(
            (self.num_units, self.model.T)
        )  # check this syntax
        return {
            "is_charging": self.is_charging,
            "is_discharging": self.is_discharging,
        }

    def define_variables(self):
        self.storage_level = cp.Variable((self.num_units, self.model.T), nonneg=True)
        self.charge_rate = cp.Variable((self.num_units, self.model.T), nonneg=True)
        self.discharge_rate = cp.Variable((self.num_units, self.model.T), nonneg=True)
        return {
            "storage_level": self.storage_level,
            "charge_rate": self.charge_rate,
            "discharge_rate": self.discharge_rate,
        }

    def define_constraints(self):
        constraints = []

        # storage level bounds
        max_levels = np.array(self.max_levels)[:, None]
        min_levels = np.array(self.min_levels)[:, None]
        constraints.append(self.storage_level <= max_levels)
        constraints.append(self.storage_level >= min_levels)

        # charge constraints
        constraints.append(
            self.charge_rate
            >= cp.multiply(self.is_charging, np.array(self.min_charge_rate)[:, None])
        )
        constraints.append(
            self.charge_rate
            <= cp.multiply(self.is_charging, np.array(self.max_charge_rate)[:, None])
        )

        # discharge constraints
        constraints.append(
            self.discharge_rate
            >= cp.multiply(
                self.is_discharging, np.array(self.min_discharge_rate)[:, None]
            )
        )
        constraints.append(
            self.discharge_rate
            <= cp.multiply(
                self.is_discharging, np.array(self.max_discharge_rate)[:, None]
            )
        )

        # Storage level constraint
        loss_factors = np.array(self.loss_factors)[:, None]
        charge_eff = np.array(self.charge_efficiencies)[:, None]
        discharge_eff = np.array(self.discharge_efficiencies)[:, None]
        initial_levels = np.array(self.initial_levels)[:, None]

        # TODO: check timestep (assumed to be 1 here)
        # for t == 0:
        constraints.append(
            self.storage_level[:, [0]]
            == initial_levels * (1 - loss_factors)
            + self.charge_rate[:, 0] * charge_eff
            - self.discharge_rate[:, 0] / discharge_eff
        )

        # for all other t
        constraints.append(
            self.storage_level[:, 1:]
            == cp.multiply((1 - loss_factors), self.storage_level[:, :-1])
            + cp.multiply(self.charge_rate[:, 1:], charge_eff)
            - cp.multiply(self.discharge_rate[:, 1:], 1 / discharge_eff)
        )

        # End storage level constraints
        min_ending = np.array(self.min_ending_levels)  # (U,)
        max_ending = np.array(self.max_ending_levels)  # (U,)
        constraints.append(self.storage_level[:, -1] >= min_ending)
        constraints.append(self.storage_level[:, -1] <= max_ending)

        return constraints

    def define_objective(self):
        charge_costs = cp.sum(self.charge_costs * self.charge_rate)  # check dims
        discharge_costs = cp.sum(
            self.discharge_costs * self.discharge_rate
        )  # check dims
        return charge_costs + discharge_costs


class ThermalGeneratorsComponent(UCComponent):
    def __init__(self, model, thermal_units_list: list[ThermalGeneratorData]):
        super().__init__(model)
        self.num_units = len(thermal_units_list)
        self.mw_cost_curves = [
            g.production_cost_curve for g in thermal_units_list
        ]  # list of lists of (mw, cost) tuples

        self.min_power = np.array(
            [g.min_power for g in thermal_units_list]
        )  # shape (n_units,)

        self.max_power = np.array([g.max_power for g in thermal_units_list])

        # min up/down times and initial status
        self.min_uptime = np.array(
            [g.min_up_time for g in thermal_units_list], dtype=int
        )
        self.min_downtime = np.array(
            [g.min_down_time for g in thermal_units_list], dtype=int
        )
        self.initial_status = np.array(
            [g.initial_status for g in thermal_units_list], dtype=int
        )
        self.max_segments = max(
            len(g.production_cost_curve) for g in thermal_units_list
        )  # number of piecewise segments

        self.segment_mw, self.segment_cost = self._build_segment_mw_cost(
            self.mw_cost_curves, self.max_segments
        )  # shape (n_units, 1, max_segments)

    def define_parameters(self):
        self.is_on = cp.Parameter((self.num_units, self.model.T))

        return {
            "is_on": self.is_on,
        }

    def define_variables(self):  # assumes no must-run units, check with the data
        self.segprod = cp.Variable(
            (self.num_units, self.model.T, self.max_segments), nonneg=True
        )  # production in each segment
        self.prod_above = cp.Variable(
            (self.num_units, self.model.T), nonneg=True
        )  # what is this for? do you need this is in the case of no reserves?

        return {
            "segprod": self.segprod,
            "prod_above": self.prod_above,
        }

    def define_constraints(self):  # chatgpt made these, double-check.
        constraints = []

        max_power = np.maximum(self.max_power, 0.0)  # shape (G, T)
        min_power = np.maximum(self.min_power, 0.0)  # shape (G, T)

        power_diff = max_power - min_power
        power_diff[power_diff < 1e-7] = 0.0

        constraints.append(
            self.prod_above <= cp.multiply(power_diff[:, None], self.is_on)
        )

        # Piecewise production cost constraints
        constraints.append(
            self.segprod <= cp.multiply(self.segment_mw, self.is_on[:, :, None])
        )

        # Optional explicit upper bound: segprod[g,t,k] <= segment_mw[g,t,k]
        constraints.append(self.segprod <= self.segment_mw)

        # prod_above[g,t] == sum_k segprod[g,t,k]
        constraints.append(self.prod_above == cp.sum(self.segprod, axis=2))

        return constraints

    def define_objective(self):

        # Segment costs
        segment_costs = cp.sum(cp.multiply(self.segprod, self.segment_cost))

        return segment_costs

    def _build_segment_mw_cost(self, curves, max_segments):
        num_units = len(curves)
        segment_mw = np.zeros((num_units, 1, max_segments))
        segment_cost = np.zeros((num_units, 1, max_segments))

        for i, seg_list in enumerate(curves):
            all_mw = [mw for mw, cost in seg_list]
            all_cost = [cost for mw, cost in seg_list]
            n = len(all_mw)
            segment_mw[i, 0, :n] = all_mw
            segment_cost[i, 0, :n] = all_cost

        return segment_mw, segment_cost


class SystemComponent(UCComponent):
    def __init__(self, model, system_data: SystemData):
        super().__init__(model)
        self.power_balance_penalty = system_data.power_balance_penalty
        self.load = system_data.load
        self.T = model.T

    def define_variables(self):
        self.curtailment = cp.Variable(self.T, nonneg=True, name="curtailment")
        return {"curtailment": self.curtailment}

    def define_parameters(self):
        self.load = cp.Parameter(self.T, name="load")
        return {"load": self.load}

    def define_constraints(self):
        constraints = []

        total_generation_thermal = cp.sum(
            self.model.thermal_gens.prod_above, axis=0
        ) + cp.sum(
            cp.multiply(
                self.model.thermal_gens.is_on,
                self.model.thermal_gens.min_power[:, None],
            ),
            axis=0,
        )  # check axis?
        total_generation_profiled = cp.sum(
            self.model.profiled_gens.profiled_generation, axis=0
        )
        total_discharging_storage = cp.sum(
            self.model.storage_units.discharge_rate, axis=0
        )
        total_charging_storage = cp.sum(self.model.storage_units.charge_rate, axis=0)
        total_generation = (
            total_generation_thermal
            + total_generation_profiled
            + total_discharging_storage
        )
        power_balance = (
            total_generation + self.curtailment == self.load + total_charging_storage
        )
        constraints.append(power_balance)

        curtailment_limits = self.curtailment <= self.load
        constraints.append(curtailment_limits)

        return constraints

    def define_objective(self):
        objective = self.power_balance_penalty * cp.sum(self.curtailment)
        return objective
