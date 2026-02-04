from src.datasets.simple_dataset import SimpleDataset
from exploratory_data_analysis_utils import plot_expected_commitment_given_load

data_dir = "data/Train_Data"
dataset = SimpleDataset(data_dir)
plot_expected_commitment_given_load(dataset, n_bins=20, max_load=8180)
