from src.datasets.simple_dataset import SimpleDataset
data_dir = "data/starting_kit_ai-uc_v2/Train_Data"
dataset = SimpleDataset(data_dir=data_dir)  # change the data_dir later

for i in range(len(dataset)):
    _  = dataset[i]
