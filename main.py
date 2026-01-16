import argparse
from datetime import datetime
import os
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


def train_epoch(model, ed_layer, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        features = batch["features"]
        targets = batch["target"]

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
        load = features["profiles"][:, :, 0]
        solar_max = features["profiles"][:, :, 2].unsqueeze(1)
        wind_max = features["profiles"][:, :, 1].unsqueeze(1)
        print("Solving LP")
        ed_solution = ed_layer(
            load,
            solar_max,
            wind_max,
            outputs_dict["is_on_rounded"],
            is_charging,
            is_discharging,
            # solver_args={"verbose": True},
        )

        ## Compute loss
        initial_commitment = features["initial_conditions"][:, :, -1] > 0
        initial_status = features["initial_conditions"][:, :, -1]
        print("Computing Loss")
        loss = criterion(
            ed_solution, outputs_dict, targets, initial_status, initial_commitment
        )

        ### Backward pass
        print("backproping")
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
        train_ds, batch_size=cfg.training.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False)

    # Instantiate model
    model_name = cfg.model.name
    ModelClass = getattr(models, model_name)
    model = ModelClass(**cfg.model.hyper_params.__dict__)

    # Instantiate ED layer
    ed_data_dict = create_data_dict(cfg.dataset.ed_instance_path)
    ed_layer = UCModel(ed_data_dict).build_layer()

    # Loss and optimizer
    criterion = getattr(losses, cfg.training.criterion)(ed_data_dict)
    optimizer = getattr(torch.optim, cfg.training.optimizer)(
        model.parameters(), lr=cfg.training.learning_rate
    )

    train_losses = []
    val_losses = []  # store (epoch, val_loss)

    val_every = getattr(cfg.training, "val_every", 5)

    for epoch in range(cfg.training.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        train_loss = train_epoch(model, ed_layer, train_loader, criterion, optimizer)
        train_losses.append(train_loss)

        # Validate only every val_every epochs, and always on the last one
        do_val = ((epoch + 1) % val_every == 0) or (
            epoch + 1 == cfg.training.num_epochs
        )

        if do_val:
            val_loss = eval_epoch(model, val_loader, criterion)
            val_losses.append((epoch + 1, val_loss))
            print(
                f"Epoch {epoch + 1}/{cfg.training.num_epochs} "
                f"- train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{cfg.training.num_epochs} "
                f"- train_loss: {train_loss:.4f}"
            )

    # Save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_save_path = "results/"
    os.makedirs(
        os.path.join(base_save_path, cfg.experiment_name, timestamp), exist_ok=True
    )

    # Save the model weights
    weights_save_path = os.path.join(
        base_save_path, cfg.experiment_name, timestamp, "simple_mlp_state.pt"
    )
    torch.save(model.state_dict(), weights_save_path)
    print(f"Model saved to {weights_save_path}")

    # Save the whole mdoel
    model_save_path = os.path.join(
        base_save_path, cfg.experiment_name, timestamp, "simple_mlp_model.pt"
    )
    torch.save(model, model_save_path)

    # Save losses
    loss_path = os.path.join(
        base_save_path, cfg.experiment_name, timestamp, "losses.pt"
    )
    torch.save({"train_losses": train_losses, "val_losses": val_losses}, loss_path)
    print(f"Losses saved to {loss_path}")

    # Save test indices # TODO: figure out some other way to do this later.
    test_indices_path = os.path.join(
        base_save_path, cfg.experiment_name, timestamp, "test_indices.pt"
    )
    torch.save(test_ds.indices, test_indices_path)
    print(f"Test indices saved to {test_indices_path}")

    # Save config
    config_save_path = os.path.join(
        base_save_path, cfg.experiment_name, timestamp, "config.yaml"
    )
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"Config saved to {config_save_path}")


if __name__ == "__main__":
    main()
