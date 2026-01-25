import argparse
from datetime import datetime
import os
import juliacall
import torch
from tqdm import tqdm
import yaml
from src.datasets.simple_dataset import SimpleDataset
from torch.utils.data import DataLoader, random_split
import src.models.simple_mlp as models
from src.models.round import ste_round
import src.utils.losses as losses
from src.models.ed_model import UCModel
from src.models.data_classes import create_data_dict
import ipdb
from src.utils.profiling import benchmark_batchsize_sweep, format_sweep_results
from torch.utils.data import Subset

MAX_BS = 64
WARMUP = 3
STEPS = 3

needed = MAX_BS * (WARMUP + STEPS)

class Config:
    def __init__(self, cfg_dict):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                value = Config(value)  # recursively convert nested dicts
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__dict__}"


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    return Config(raw_cfg)


def train_epoch(model, ed_layer, dataloader, criterion, optimizer, device=torch.device("cpu")):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):

        features = {k: v.to(device) for k, v in batch["features"].items()}
        targets = {k: v.to(device) for k, v in batch["target"].items()} if isinstance(batch["target"], dict) else batch["target"].to(device)


        ### Forward pass

        ## Get NN output (relaxed commitments)
        outputs_dict = model(features)
        is_charging = outputs_dict["is_charging"]  # TODO: check dims
        is_discharging = outputs_dict["is_discharging"]  # TODO: check dims

        ## Apply rounding if specified
        outputs_dict["is_on_rounded"] = ste_round(
            outputs_dict["is_on"]
        )  # TODO: check dims

        ## Solve LP
        ### Solve ED problem given commitments and features
        load = features["profiles"][:, :, 0].to(device)
        solar_max = features["profiles"][:, :, 2].unsqueeze(1).to(device)
        wind_max = features["profiles"][:, :, 1].unsqueeze(1).to(device)
        ed_solution = ed_layer(
            load,
            solar_max,
            wind_max,
            outputs_dict["is_on_rounded"],
            is_charging,
            is_discharging,
            solver_args={"max_iters": 10000},
        )

        ## Compute loss
        initial_commitment = features["initial_conditions"][:, :, -1] > 0
        initial_status = features["initial_conditions"][:, :, -1]
        loss = criterion(
            ed_solution, outputs_dict, targets, initial_status, initial_commitment
        )

        ### Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    for batch in dataloader:
        features = batch["features"]
        targets = batch["target"]
        outputs = model(features)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
    return total_loss / len(dataloader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple MLP on the dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/idea_1_config.yaml",
        help="Path to the configuration file (YAML format).",
    )
    args = parser.parse_args()


    # Load config file
    cfg = load_config(args.config)

    # Set up device
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.device == "cuda") else "cpu")
    print(f"Using device: {device}")

    # Set up cvxpylayers
    backend = cfg.backend
    solver = cfg.solver

    # Set seeds for reproducibility
    seed = cfg.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize dataset
    dataset = SimpleDataset(data_dir=cfg.dataset.data_dir)  # change the data_dir later.

    # Compute split sizes
    n_total = len(dataset)
    n_train = int(cfg.splits.train * n_total)
    n_val = int(cfg.splits.val * n_total)
    n_test = n_total - n_train - n_val

    # DataLoaders
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)

    # Instantiate model
    model_name = cfg.model.name
    ModelClass = getattr(models, model_name)
    model = ModelClass(**cfg.model.hyper_params.__dict__).to(device)

    # Instantiate ED layer
    ed_data_dict = create_data_dict(cfg.dataset.ed_instance_path)
    ed_layer = UCModel(ed_data_dict).build_layer(device=device, backend=backend, solver=solver)

    # Loss and optimizer
    criterion = getattr(losses, cfg.training.criterion)(ed_data_dict)
    optimizer = getattr(torch.optim, cfg.training.optimizer)(
        model.parameters(), lr=cfg.training.learning_rate
    )

    train_losses = []
    val_losses = []  # store (epoch, val_loss)

    results = benchmark_batchsize_sweep(
        model=model,
        ed_layer=ed_layer,
        dataset=dataset,
        criterion=criterion,
        device=device,
        ste_round_fn=ste_round,
        batch_sizes=[1, 2, 4, 8],
        steps=3,          # spread over 3 measured steps
        warmup=3,         # ignore warmup
        solver_args={"max_iters": 10000},
        drop_last=True,
    )

    print(format_sweep_results(results))


if __name__ == "__main__":
    main()
