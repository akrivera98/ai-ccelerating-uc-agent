import cvxpy as cp
from src.models.components import (
    ProfiledGeneratorsComponent,
    SystemComponent,
    StorageUnitsComponent,
    ThermalGeneratorsComponent,
)


class UCModel:
    def __init__(self, data_dict):
        """
        profiled_gen_data_list: list of ProfiledGeneratorData
        load_profile: np.array of shape (T,)   (system load per time step)
        """
        self.T = len(data_dict["system_data"].load)

        # Components
        self.profiled_gens = ProfiledGeneratorsComponent(
            self, data_dict["profiled_gen_data_list"]
        )
        self.system = SystemComponent(self, data_dict["system_data"])
        self.storage_units = StorageUnitsComponent(self, data_dict["storage_data_list"])
        self.thermal_gens = ThermalGeneratorsComponent(
            self, data_dict["thermal_gen_data_list"]
        )

        # Hold cvxpy objects
        self.variables = {}
        self.parameters = {}
        self.constraints = []
        self.objective = None
        self.problem = None

    def build(self):
        # 1. Define variables for each component
        self.variables.update(self.profiled_gens.define_variables())
        self.variables.update(self.storage_units.define_variables())
        self.variables.update(self.thermal_gens.define_variables())
        self.variables.update(self.system.define_variables())

        # 2. Define parameters for each component
        self.parameters.update(self.storage_units.define_parameters())
        self.parameters.update(self.thermal_gens.define_parameters())

        # 2. Collect constraints from components
        self.constraints += self.profiled_gens.define_constraints()
        self.constraints += self.system.define_constraints()
        self.constraints += self.storage_units.define_constraints()
        self.constraints += self.thermal_gens.define_constraints()

        # 4. Objective: minimize total cost
        profiled_gen_cost = self.profiled_gens.define_objective()
        system_cost = self.system.define_objective()
        storage_cost = self.storage_units.define_objective()
        thermal_gen_cost = self.thermal_gens.define_objective()
        total_cost = profiled_gen_cost + system_cost + storage_cost + thermal_gen_cost
        self.objective = cp.Minimize(total_cost)

        # 5. Build the CVXPY problem
        self.problem = cp.Problem(self.objective, self.constraints)

    def solve(self, **kwargs):
        if self.problem is None:
            self.build()
        return self.problem.solve(**kwargs)
