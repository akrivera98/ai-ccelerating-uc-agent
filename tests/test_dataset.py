import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.simple_dataset import SimpleDataset

data_dir = "data/starting_kit/Train_Data"
dataset = SimpleDataset(data_dir=data_dir)
print(f"Dataset length: {len(dataset)}")
sample = dataset[0]
print(f"Sample features shape: {sample['features'].shape}")
print(f"Sample target shape: {sample['target'].shape}")
print(f"Sample features: {sample['features']}")
print(f"Sample target: {sample['target']}")
