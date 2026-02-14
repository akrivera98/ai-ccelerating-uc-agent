from src.datasets.uc_dataset import UCDataset

data_dir = "data/starting_kit_ai-uc_v2/Train_Data"
dataset = UCDataset(data_dir=data_dir)  # change the data_dir later

for i in range(len(dataset)):
    _ = dataset[i]
