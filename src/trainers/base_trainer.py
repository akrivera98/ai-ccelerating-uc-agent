from abc import ABC, abstractmethod
import json
import random
from time import time
import torch
from src.registry import registry
from torch.utils.data import random_split
import numpy as np
import os


class BaseTrainer(ABC):
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.device = config.device
        self.is_debug = config.is_debug

        # training
        self.max_epochs = config.trainer.max_epochs
        self.optimizer = config.trainer.optimizer.name
        self.batch_size = config.trainer.batch_size

        # validation
        self.do_val = config.trainer.do_val
        self.val_every = config.trainer.val_every

        # dataset
        self.train_split = config.dataset.splits.train
        self.val_split = config.dataset.splits.val
        self.test_split = config.dataset.splits.test
        self.dataset_size = getattr(config.dataset, "subset", None)

        # stopping criteria
        self.early_stopping_patience = getattr(
            config.trainer, "early_stopping_patience", None
        )

        self.early_stopping_perc = getattr(config.trainer, "early_stopping_perc", 0.05)
        self._es_best = None

        # datasets and dataloader
        self.data_dir = config.dataset.params.data_dir
        self.train_split = config.dataset.splits.train
        self.val_split = config.dataset.splits.val

        # tracking
        self.train_errors = {}
        self.val_errors = {}
        self.best_point = None
        self.best_model = None
        self._es_counter = 0

        # create output directory and save config
        self.save_config()
        out_dir = self.config.paths.out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.setup_training()

    def setup_training(self):
        self.load_seed()
        self.load_dataset()
        self.load_dataloaders()
        self.load_model()
        self.load_ed_model()
        self.load_loss_fn()
        self.load_optimizer()

    def load_dataset(self):
        dataset_name = self.config.dataset.name
        dataset_params = self.config.dataset.params
        dataset_class = registry.get_dataset(dataset_name)
        self.dataset = dataset_class(**dataset_params.to_dict())
        if self.dataset_size is not None:
            self.dataset = torch.utils.data.Subset(
                self.dataset, list(range(self.dataset_size))
            )

    def load_dataloaders(self):
        assert self.train_split + self.val_split + self.test_split == 1.0, (
            "Train, val, and test splits must sum to 1"
        )
        train_ds, val_ds, test_ds = random_split(
            self.dataset,
            [self.train_split, self.val_split, self.test_split],
            generator=self.generator,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        if self.do_val:
            self.val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False
            )

        # test_ds indices for later evaluation
        self.test_indices = test_ds.indices

    def load_optimizer(self):
        optimizer_class = getattr(torch.optim, self.optimizer)
        optimizer_params = self.config.trainer.optimizer.params.to_dict()
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)

    def load_seed(self):
        seed = self.config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        self.generator = g

    def load_model(self):
        model_name = self.config.model.name
        model_params = self.config.model.params
        model_class = registry.get_model(model_name)
        self.model = model_class(**model_params.to_dict()).to(self.device)

    def load_ed_model(self):
        ed_model_name = getattr(self.config.ed_model, "name", None)
        if ed_model_name is not None:
            ed_model_params = self.config.ed_model.params
            ed_model_class = registry.get_ed_model(ed_model_name)
            self.ed_model = ed_model_class(**ed_model_params.to_dict())
        else:
            self.ed_model = None

    def load_loss_fn(self):
        loss_fn_name = self.config.trainer.loss_fn.name
        loss_fn_params = self.config.trainer.loss_fn.params
        loss_fn_class = registry.get_loss(loss_fn_name)
        self.loss_fn = loss_fn_class(**loss_fn_params.to_dict()).to(self.device)
        self.loss_names = self.loss_fn.loss_names

    def save(self, *, stage, train_loss, val_loss=None):
        out_dir = self.config.paths.out_dir
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        pass

    def save_config(self):
        out_dir = self.config.paths.out_dir
        os.makedirs(out_dir, exist_ok=True)
        config_location = os.path.join(out_dir, "config.yaml")
        self.config.save_yaml(config_location)

    def train(self):

        start = time()

        # Start training
        train_dataloader = self.train_loader
        self.epoch = 0
        while True:
            print(
                f"\nEpoch {self.epoch + 1}/{self.max_epochs if self.max_epochs is not None else 'inf'}"
            )
            self.model.train()

            # Train one epoch and gather losses
            running_losses = self.train_epoch()
            losses = [loss / len(train_dataloader) for loss in running_losses.values()]

            # Report and save results
            self.report_results(losses)

            # Calc val error and save new best model if necessary
            if (
                self.do_val
                and (self.val_every is not None)
                and ((self.epoch + 1) % self.val_every == 0)
            ):
                self.compute_val_error()

                # Early stopping condition
                stop_now = self.early_stop_now()  # TODO: implement
                if stop_now:
                    print("\n## Early stop triggered...")
                    break

            # End training condition
            if self.epoch >= self.max_epochs - 1:
                break

            self.epoch += 1

        print("Training done")
        print("Calculating final validation error...")
        self.compute_val_error(last_time=True)
        end = time()
        print(f"Training finished in {(end - start):.2f} seconds.")
        print()

    def early_stop_now(self):
        if self.early_stopping_patience is None:
            return False

        # first observation
        if self._es_best is None:
            self._es_best = self.val_errors[str(self.epoch)]["total"]
            return False

        # check if val_loss is better than best by at least percentage
        if self.val_errors[str(self.epoch)]["total"] < self._es_best * (
            1 - self.early_stopping_perc
        ):
            self._es_best = self.val_errors[str(self.epoch)]["total"]
            self._es_counter = 0
            return False
        else:
            self._es_counter += 1
            if self._es_counter >= self.early_stopping_patience:
                return True
            else:
                return False

    def save_model(self):

        first_time = self.best_point is None
        is_new_best, best_perf = self.is_new_best()

        model_save_path = os.path.join(self.config.paths.out_dir, "model.pt")

        if is_new_best:
            if not self.is_debug:
                torch.save(self.model.state_dict(), model_save_path)
            if not first_time:
                print(
                    f"New best model saved with validation performance: {best_perf:.4f}"
                )
            self.best_point = self.epoch

    def is_new_best(self):
        last_value = self.val_errors[str(self.epoch)]["total"]  # should be total loss
        old_best = (
            self.val_errors[str(self.best_point)]["total"]
            if self.best_point is not None
            else float("inf")
        )
        return last_value < old_best, last_value

    def save_summary(self, last_time=False):
        val_errors = self.val_errors[str(self.best_point)]
        train_errors = self.train_errors[str(self.best_point)]

        # Save results
        run_location = self.config.paths.out_dir
        summary_location = os.path.join(run_location, "summary.json")

        summary = {
            "train": train_errors,
            "val": val_errors,
            "best_point": self.best_point,
            "finished": last_time,
        }

        if not self.is_debug:
            with open(summary_location, "w") as f:
                json.dump(summary, f, indent=4)

    def report_results(self, losses):
        current_point = self.epoch

        self.train_errors[str(current_point)] = {}
        for i, loss_name in enumerate(self.loss_names):
            self.train_errors[str(current_point)][loss_name] = losses[i]

        train_location = os.path.join(self.config.paths.out_dir, "train.json")
        if not self.is_debug:
            with open(train_location, "w") as f:
                json.dump(self.train_errors, f, indent=4)

    def compute_val_error(self, last_time=False):
        val_loader = self.val_loader

        current_point = self.epoch

        print("Calculating validation error...")
        running_losses = self.eval_batch()
        losses = [loss / len(val_loader) for loss in running_losses.values()]
        self.val_errors[str(current_point)] = {}
        for i, loss_name in enumerate(self.loss_names):
            self.val_errors[str(current_point)][loss_name] = losses[i]

        val_location = os.path.join(self.config.paths.out_dir, "val.json")
        if not self.is_debug:
            with open(val_location, "w") as f:
                json.dump(self.val_errors, f, indent=4)

        self.save_model()
        self.save_summary(last_time)

    @abstractmethod
    def train_epoch(self):
        """
        Docstring for train_epoch

        :param self: Description

        return: running_losses
        """
        return NotImplementedError

    @abstractmethod
    def eval_batch(self):
        pass
