from src.utils.losses import _evaluate_uptime_downtime_constraints
from src.datasets.simple_dataset import SimpleDataset
import torch
dataset = SimpleDataset(data_dir="data/starting_kit/Train_Data")[0]  # change the data_dir later.
targets = dataset["target"]
dummy_min_uptimes = torch.randint(1, 5, (51,))  # assuming 51 thermal generators

violations = _evaluate_uptime_downtime_constraints(
    outputs=targets,
    min_uptimes=dummy_min_uptimes,
    min_downtimes=dummy_min_uptimes,
    initial_status=torch.zeros(51)
)
