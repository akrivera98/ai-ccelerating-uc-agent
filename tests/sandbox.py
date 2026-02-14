from src.datasets.uc_dataset import UCDataset

data_dir = "data/Train_Data"

dataset = UCDataset(data_dir)

print(dataset[0])
