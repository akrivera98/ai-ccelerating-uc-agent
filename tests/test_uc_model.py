import numpy as np
from src.models.ed_model import UCModel
from src.ed_models.data_utils import create_data_dict

sample_path = "sample.json"
data_dict = create_data_dict(sample_path)

uc_model = UCModel(data_dict)
uc_model.build()

T = uc_model.T
num_units = uc_model.thermal_gens.num_units
num_storage = uc_model.storage_units.num_units

# --- Thermal generator parameters ---

# For debugging / smoke tests, just turn everything ON
is_on_val = np.ones((num_units, T), dtype=bool)
is_charging_val = np.zeros((num_storage, T), dtype=bool)
is_discharging_val = np.ones((num_storage, T), dtype=bool)

uc_model.thermal_gens.is_on.value = is_on_val
uc_model.storage_units.is_charging.value = is_charging_val
uc_model.storage_units.is_discharging.value = is_discharging_val

# --- Solve ---
solution = uc_model.solve()
print("Objective value:", solution)
print("Problem status:", uc_model.problem.status)
