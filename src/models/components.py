from dataclasses import dataclass
from typing import Tuple
from abc import ABC, abstractmethod
import cvxpy as cp

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



# Component Data
@dataclass
class ThermalGeneratorData:
    name: str
    production_cost_curve: list[Tuple[float, float]] # List of ($, MW) pairs
    startup_costs: list[float]
    startup_delays: list[float] # or int # double-check how this will be used
    ramp_up_limit: float
    ramp_down_limit: float
    startup_limit: float
    shutdown_limit: float
    min_down_time: float
    min_up_time: float
    initial_power: float
    initial_status: float
    commitment_status: bool

@dataclass
class ProfiledGeneratorData:
    name: str
    cost: float
    min_power: float
    max_power: float

@dataclass
class SystemData:
    power_balance_penalty: float
    load: list[float]


class SystemComponent(UCComponent):
    def __init__(self, model, system_data: SystemData):
        super().__init__(model)
        self.power_balance_penalty = system_data.power_balance_penalty
        self.load = system_data.load

    def define_variables(self):
        self.curtailment = cp.Variable(T, nonneg=True, name="curtailment")

    def define_parameters(self):
        self.load = cp.Parameter(T, name="load")

    def define_constraints(self):
        constraints = []

        total_generation_thermal = cp.sum(self.thermal_generation, axis=0)
        total_generation_profiled = cp.sum(self.profiled_generation, axis=0)
        total_generation = total_generation_thermal + total_generation_profiled
        power_balance = total_generation + self.curtailment == self.load
        constraints.append(power_balance)
        return constraints
    
    def define_objective(self):
        objective = self.power_balance_penalty * cp.sum(self.curtailment)
        return objective
    
class ProfiledGeneratorComponent(UCComponent):
    def __init__(self, model, profiled_gen_data: ProfiledGeneratorData):
        super().__init__(model)
        self.cost = profiled_gen_data.cost
        self.min_power = profiled_gen_data.min_power
        self.max_power = profiled_gen_data.max_power
    
    def define_variables(self):
        self.generation = cp.Variable(T, name=f"generation_{self.name}") # Note that you're not making multi D arrays here.
        return {"generation": self.generation}

class ThermalGeneratorComponent(UCComponent):
    def __init__(self, model, thermal_gen_data: ThermalGeneratorData):
        super().__init__(model)
        self.gen = thermal_gen_data
    
    def define_variables():
        pass


