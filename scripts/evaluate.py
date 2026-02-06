import argparse

import torch
from src.datasets.simple_dataset import SimpleDataset
from torch.utils.data import Subset, DataLoader
from src.models.ed_model_qp import EDModelLP
from src.models.data_classes import create_data_dict
from src.models.fnn import TwoHeadMLP
from src.utils.evaluation import run_standard_evaluation
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the data directory"
    )
    parser.add_argument(
        "--test-indices", type=str, required=True, help="Path to the test indices file"
    )
    parser.add_argument(
        "--ed-instance-path",
        type=str,
        required=True,
        help="Path to the ED instance data file",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the evaluation on"
    )
    parser.add_argument(
        "--save-path", type=str, default=None, help="Path to save evaluation results"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained model file"
    )
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the model config file"
    ) 
    return parser.parse_args()


def load_test_indices(path):
    test_indices = torch.load(path)
    return test_indices

def get_model_params_from_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model_params = config['model']['hyper_params']
    input_size = model_params['input_size']
    hidden_size = model_params['hidden_size']
    num_hidden_layers = model_params['num_hidden_layers']
    return input_size, hidden_size, num_hidden_layers

def main():
    # Parse arguments
    args = parse_args()

    # Build test dataset
    full_ds = SimpleDataset(args.data_dir)
    test_indices = load_test_indices(args.test_indices)
    test_ds = Subset(full_ds, test_indices)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    ed_data_dict = create_data_dict(args.ed_instance_path)

    # Build ED/LP layer
    ed_layer = EDModelLP(ed_data_dict).to(device=args.device)

    # Load model
    input_size, hidden_size, num_hidden_layers = get_model_params_from_config(args.config_path)
    model = TwoHeadMLP(
        input_size=input_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=args.device)) 

    # Run evaluation and save results
    results = run_standard_evaluation(
        model, ed_layer, test_loader, args.device, save_path=args.save_path
    )

if __name__ == "__main__":
    main()
