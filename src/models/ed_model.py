import torch.nn as nn
from cvxpylayers.torch import CvxpyLayer
from abc import ABC, abstractmethod
import cvxpy as cp
from dataclasses import dataclass
from typing import Tuple

class EconomicDispatchModel:
    def __init__(self, system_data):
        self.data = system_data

        # Registries
        self.variables = {} # name -> cvxpy Variable
        self.constraints = [] # list of cvxpy Constraints
        self.objective_terms = [] # list of cvxpy Expressions

        # Index maps
        self.gen_thermal_idx = {}
        self.gen_profiled_idx = {}
        self.sto_idx = {}

    def build(self):
        self.add_thermal_generators()
        self.add_profiled_generators()
        self.add_storage_units()
        self.add_system_constraints()
        return self.build_problem()
    
    def build_problem(self):
        objective = None
        return cp.Problem(objective, self.constraints)


