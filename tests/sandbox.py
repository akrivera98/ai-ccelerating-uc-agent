from src.datasets.simple_dataset import SimpleDataset

data_dir = "data/Train_Data"

dataset = SimpleDataset(data_dir)

print(dataset[0])