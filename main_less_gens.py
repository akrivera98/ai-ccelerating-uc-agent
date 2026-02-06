import argparse
from datetime import datetime
import os
import torch
from tqdm import tqdm
import yaml
from src.datasets.simple_dataset import SimpleDataset
from torch.utils.data import DataLoader, random_split
import src.models.fnn as models
from src.models.ed_model_qp import EDModelLP
from src.models.data_classes import create_data_dict
from collections import defaultdict
from src.utils.losses import CustomLoss
import pickle


def save_checkpoint(
    model,
    optimizer,
    epoch,
    cfg,
    base_save_path,
    timestamp,
    train_losses,
    train_traces,
    val_losses,
):
    ckpt_dir = os.path.join(
        base_save_path, cfg.experiment_name, timestamp, "checkpoints"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "train_traces": train_traces,
            "val_losses": val_losses,
            "config": cfg,
        },
        ckpt_path,
    )

    print(f"Checkpoint saved: {ckpt_path}")


class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        mode: str = "min",
        min_rel_improve: float | None = None,
        eps: float = 1e-8,
    ):
        """
        min_delta: absolute improvement required
        min_rel_improve: relative improvement required (e.g. 0.005 = 0.5%)
        Use one or both. If both provided, improvement must satisfy BOTH.
        """
        assert mode in ("min", "max")
        self.patience = int(patience)
        self.mode = mode
        self.min_rel_improve = (
            None if min_rel_improve is None else float(min_rel_improve)
        )
        self.eps = float(eps)

        self.best = None
        self.num_bad = 0

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True

        if self.mode == "min":
            abs_improve = self.best - value
            rel_improve = abs_improve / max(abs(self.best), self.eps)
        else:
            abs_improve = value - self.best
            rel_improve = abs_improve / max(abs(self.best), self.eps)

        ok_rel = (
            True
            if self.min_rel_improve is None
            else (rel_improve >= self.min_rel_improve)
        )

        return ok_rel

    def step(self, value: float) -> tuple[bool, bool]:
        """
        Returns: (should_stop, is_new_best)
        """
        if self._improved(value):
            self.best = value
            self.num_bad = 0
            return False, True
        else:
            self.num_bad += 1
            return (self.num_bad >= self.patience), False


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


def save_results():
    pass


def validate_config(cfg: dict) -> None:
    cfg.training.learning_rate = float(cfg.training.learning_rate)

    # If no rounding for training, can't use ED in training and can't evaluate violations or startup loss
    if not cfg.ablation.rounding.train:
        cfg.ablation.use_ed_in_training = False
        cfg.ablation.loss_weights.eval_violations = 0.0
        cfg.ablation.loss_weights.startup_loss = 0.0
        cfg.ablation.loss_weights.ed_objective = 0.0


def train_epoch(
    model,
    ed_layer,
    dataloader,
    criterion,
    optimizer,
    ablations_settings=None,
    device=torch.device("cpu"),
):
    model.train()
    total_loss = 0
    n = 0
    traces = defaultdict(list)

    if ablations_settings is None:  # default settings
        ablations_settings = Config({})
        ablations_settings.rounding = Config({"train": False})
        ablations_settings.use_ed_in_training = False
        ablations_settings.loss_weights = Config({})

    for batch in tqdm(dataloader):
        features = {k: v.to(device) for k, v in batch["features"].items()}
        targets = (
            {k: v.to(device) for k, v in batch["target"].items()}
            if isinstance(batch["target"], dict)
            else batch["target"].to(device)
        )

        ### Forward pass

        ## Get NN output
        outputs_dict = model(features)

        ## Solve LP and evaluate loss
        load = features["profiles"][:, :, 0].to(device)
        wind_max = features["profiles"][:, :, 1].to(device)
        solar_max = features["profiles"][:, :, 2].to(device)

        ## Compute loss
        initial_commitment = features["initial_conditions"][:, -1, :] > 0
        initial_status = features["initial_conditions"][:, -1, :]
        loss_dict = criterion(
            ed_layer,
            outputs_dict,
            targets,
            initial_status,
            initial_commitment,
            load,
            solar_max,
            wind_max,
        )
        loss = loss_dict["total"]

        ### Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # keep track of the loss
        bs = load.shape[0]
        total_loss += loss.item() * bs  # this feels weird
        n += 1

        # save all components each step
        for k, v in loss_dict.items():
            traces[k].append(v.detach().item())

    mean_loss = total_loss / max(n, 1)

    return mean_loss, dict(traces)


@torch.no_grad()
def eval_epoch(model, ed_layer, dataloader, criterion, device=torch.device("cpu")):
    model.eval()
    total_loss = 0
    n = 0
    traces = defaultdict(list)
    for batch in dataloader:
        features = {k: v.to(device) for k, v in batch["features"].items()}
        targets = (
            {k: v.to(device) for k, v in batch["target"].items()}
            if isinstance(batch["target"], dict)
            else batch["target"].to(device)
        )

        outputs_dict = model(features)

        load = features["profiles"][:, :, 0].to(device)
        wind_max = features["profiles"][:, :, 1].to(device)
        solar_max = features["profiles"][:, :, 2].to(device)

        initial_commitment = features["initial_conditions"][:, -1, :] > 0
        initial_status = features["initial_conditions"][:, -1, :]

        loss_dict = criterion(
            ed_layer,
            outputs_dict,
            targets,
            initial_status,
            initial_commitment,
            load,
            solar_max,
            wind_max,
        )

        bs = load.shape[0]
        n += 1
        total_loss += loss_dict["total"].item() * bs

        # save all components each step
        for k, v in loss_dict.items():
            traces[k].append(v.detach().item())

    return total_loss / max(n, 1), dict(traces)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple MLP on the dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/less_gens_config.yaml",
        help="Path to the configuration file (YAML format).",
    )

    # ---- Optional sweep overrides (all optional) ----
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    # toggles
    parser.add_argument("--use-ste", type=int, default=None)  # 1/0
    parser.add_argument("--use-ed-in-training", type=int, default=None)  # 1/0

    # loss weights
    parser.add_argument("--w-supervised", type=float, default=None)
    parser.add_argument("--w-violation", type=float, default=None)
    parser.add_argument("--w-ed-objective", type=float, default=None)
    parser.add_argument("--w-startup", type=float, default=None)

    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)

    # ---- Apply CLI overrides (if provided) ----
    if args.run_name is not None:
        cfg.experiment.name = args.run_name

    if args.seed is not None:
        cfg.seed = args.seed

    if args.use_ste is not None:
        cfg.ablation.rounding.train = True if args.use_ste == 1 else False

    if args.use_ed_in_training is not None:
        cfg.ablation.use_ed_in_training = args.use_ed_in_training == 1

    # loss weights (only override if provided)
    if args.w_supervised is not None:
        cfg.ablation.loss_weights.supervised = args.w_supervised
    if args.w_violation is not None:
        cfg.ablation.loss_weights.violation = args.w_violation
    if args.w_ed_objective is not None:
        cfg.ablation.loss_weights.ed_objective = args.w_ed_objective
    if args.w_startup is not None:
        cfg.ablation.loss_weights.startup = args.w_startup

    # Validate config
    validate_config(cfg)

    # Set up device
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and cfg.device == "cuda") else "cpu"
    )
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    seed = cfg.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize dataset
    dataset = SimpleDataset(data_dir=cfg.dataset.data_dir)

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
        train_ds, batch_size=cfg.dataloader.batch_size, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.dataloader.batch_size, shuffle=False)

    # Set up model hyperparameters
    hyper_params = cfg.model.hyper_params.__dict__
    hyper_params["gen_names_all"] = dataset[0]["gen_names"]
    hyper_params["gen_names"] = pickle.load(open(cfg.model.gen_names_path, "rb"))

    # Instantiate model
    model_name = cfg.model.name
    ModelClass = getattr(models, model_name)
    model = ModelClass(**hyper_params).to(device)
    gen_idx = model.gen_idx

    # Instantiate ED layer
    ed_data_dict = create_data_dict(cfg.dataset.ed_instance_path)
    ed_layer = EDModelLP(ed_data_dict)

    # Loss and optimizer
    criterion = CustomLoss(
        ed_data_dict,
        loss_weights=cfg.ablation.loss_weights,
        solve_lp_in_loss=cfg.ablation.use_ed_in_training,
        predicting_storage=cfg.model.hyper_params.predict_storage,
        gen_idx=gen_idx,
    )
    optimizer = getattr(torch.optim, cfg.training.optimizer)(
        model.parameters(), lr=cfg.training.learning_rate
    )

    train_losses = []
    train_traces = []
    val_losses = []  # store (epoch, val_loss)

    val_every = getattr(cfg.training, "val_every", 5)
    save_every = getattr(cfg.output, "save_every", 0)

    # Outputs setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_save_path = cfg.output.base_dir
    os.makedirs(
        os.path.join(base_save_path, cfg.experiment.name, timestamp), exist_ok=True
    )

    use_es = cfg.training.early_stopping.enabled
    early_stopper = None
    best_ckpt_path = os.path.join(
        base_save_path, cfg.experiment.name, timestamp, "checkpoints", "best.pt"
    )

    if use_es:
        early_stopper = EarlyStopping(
            patience=cfg.training.early_stopping.patience,
            mode="min",
            min_rel_improve=cfg.training.early_stopping.min_rel_improve,
            eps=cfg.training.early_stopping.eps,
        )

    for epoch in range(cfg.training.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        train_loss, train_trace = train_epoch(
            model,
            ed_layer,
            train_loader,
            criterion,
            optimizer,
            ablations_settings=cfg.ablation,
            device=device,
        )
        train_losses.append(train_loss)
        train_traces.append(train_trace)

        # Validate only every val_every epochs, and always on the last one
        if val_every > 0:
            do_val = ((epoch + 1) % val_every == 0) or (
                epoch + 1 == cfg.training.num_epochs
            )
        else:
            do_val = False

        if do_val:
            val_loss, _ = eval_epoch(model, ed_layer, val_loader, criterion)
            val_losses.append((epoch + 1, val_loss))
            print(
                f"Epoch {epoch + 1}/{cfg.training.num_epochs} "
                f"- train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}"
            )

            if use_es:
                should_stop, is_new_best = early_stopper.step(val_loss)
                if is_new_best:
                    # Save best model checkpoint
                    os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                    torch.save(model.state_dict(), best_ckpt_path)
                    print(f"New best model saved to {best_ckpt_path}")
                if should_stop:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"No improvement for {early_stopper.patience} validations."
                    )
                    break

        else:
            print(
                f"Epoch {epoch + 1}/{cfg.training.num_epochs} "
                f"- train_loss: {train_loss:.4f}"
                f"- supervised_loss: {train_trace['supervised'][-1]:.4f}"
                f"- violation_loss: {train_trace['violations'][-1]:.4f}"
                f"- ed_objective_loss: {train_trace['ed'][-1]:.4f}"
                f"- startup_loss: {train_trace['startup'][-1]:.4f}"
            )  # TODO: print each component of the loss as well.

        if save_every > 0:
            if (epoch + 1) % save_every == 0:
                # TODO: add the losses trace here too.
                # Save intermediate model
                intermediate_save_path = os.path.join(
                    base_save_path,
                    cfg.experiment.name,
                    timestamp,
                    f"simple_mlp_epoch_{epoch + 1}.pt",
                )
                torch.save(
                    {
                        "train_losses": train_losses,
                        "train_traces": train_traces,
                        "val_losses": val_losses,
                    },
                    intermediate_save_path,
                )
                print(f"Intermediate losses saved to {intermediate_save_path}")

    # Save the model weights
    weights_save_path = os.path.join(
        base_save_path, cfg.experiment.name, timestamp, "simple_mlp_state.pt"
    )
    torch.save(model.state_dict(), weights_save_path)
    print(f"Model saved to {weights_save_path}")

    # Save the whole model
    model_save_path = os.path.join(
        base_save_path, cfg.experiment.name, timestamp, "simple_mlp_model.pt"
    )
    torch.save(model, model_save_path)

    # Save losses
    loss_path = os.path.join(
        base_save_path, cfg.experiment.name, timestamp, "losses.pt"
    )
    torch.save(
        {
            "train_losses": train_losses,
            "train_traces": train_traces,
            "val_losses": val_losses,
        },
        loss_path,
    )
    print(f"Losses saved to {loss_path}")

    # Save test indices # TODO: figure out some other way to do this later.
    test_indices_path = os.path.join(
        base_save_path, cfg.experiment.name, timestamp, "test_indices.pt"
    )
    torch.save(test_ds.indices, test_indices_path)
    print(f"Test indices saved to {test_indices_path}")

    # Save config
    config_save_path = os.path.join(
        base_save_path, cfg.experiment.name, timestamp, "config.yaml"
    )
    with open(config_save_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"Config saved to {config_save_path}")


if __name__ == "__main__":
    main()
