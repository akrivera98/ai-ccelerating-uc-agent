import torch
from src.models.ed_model import UCModel
from src.ed_models.data_utils import create_data_dict

sample_path = "sample.json"
data_dict = create_data_dict(sample_path)

uc_model = UCModel(data_dict)
uc_model.build()
layer = uc_model.build_layer()

T = uc_model.T
num_units = uc_model.thermal_gens.num_units
num_storage = uc_model.storage_units.num_units

# Relaxed 0/1 as floats
is_on_t = torch.ones((num_units, T), dtype=torch.double, requires_grad=True)
is_charging_t = torch.zeros((num_storage, T), dtype=torch.double, requires_grad=True)
is_discharging_t = torch.ones((num_storage, T), dtype=torch.double, requires_grad=True)

solution = layer(is_on_t, is_charging_t, is_discharging_t)

# unpack if you want
solution_dict = {name: sol for name, sol in zip(uc_model.variables.keys(), solution)}

loss = solution_dict["curtailment"].sum()  # ez example
loss.backward()

pass
