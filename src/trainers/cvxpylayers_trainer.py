import torch
from tqdm import tqdm
from src.trainers.base_trainer import BaseTrainer
from src.registry import registry


@registry.register_trainer("cvxpylayers_trainer")
class CvxpyLayersTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.ed_model_name = self.config.ed_model.name
        self.ed_model_params = self.config.ed_model.params
        self.build_ed_layer()

    def build_ed_layer(self):
        self.ed_layer = self.ed_model.build_layer()

    def train_epoch(self):
        self.model.train()
        running_loss = {k: 0.0 for k in self.loss_fn.loss_names}

        for batch in tqdm(self.train_loader):
            features = {k: v.to(self.device) for k, v in batch["features"].items()}
            targets = {k: v.to(self.device) for k, v in batch["target"].items()}

            # NN forward pass -> relaxed commitments
            outputs_dict = self.model(features)

            # Unpack outputs
            is_on = outputs_dict["is_on"]
            is_charging = outputs_dict["is_charging"]
            is_discharging = outputs_dict["is_discharging"]

            # Unpack inputs
            load = features["profiles"][:, :, 0]
            wind_max = features["profiles"][:, :, 1]
            solar_max = features["profiles"][:, :, 2]

            # Solve ED problem with cvxpylayers
            ed_solution = self.ed_layer(
                load,
                solar_max,
                wind_max,
                is_on.transpose(1, 2),
                is_charging.transpose(1, 2),
                is_discharging.transpose(1, 2),
            )  # TODO: get rid of transpose later when you update UC model

            # Compute loss
            losses = self.loss_fn(features, targets, outputs_dict, ed_solution)

            for k, v in losses.items():
                running_loss[k] += v.item()

            # Backpropagate
            losses["total"].backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return running_loss

    def eval_batch(self):
        self.model.eval()
        
        running_loss = {k: 0.0 for k in self.loss_fn.loss_names}

        with torch.inference_mode():
            for batch in tqdm(self.val_loader):
                features = {k: v.to(self.device) for k, v in batch["features"].items()}
                targets = {k: v.to(self.device) for k, v in batch["target"].items()}

                # NN forward pass -> relaxed commitments
                outputs_dict = self.model(features)

                # Unpack outputs
                is_on = outputs_dict["is_on"]
                is_charging = outputs_dict["is_charging"]
                is_discharging = outputs_dict["is_discharging"]

                # Unpack inputs
                load = features["profiles"][:, :, 0]
                wind_max = features["profiles"][:, :, 1]
                solar_max = features["profiles"][:, :, 2]

                # Solve ED problem with cvxpylayers
                ed_solution = self.ed_layer(
                    load,
                    solar_max,
                    wind_max,
                    is_on.transpose(1, 2),
                    is_charging.transpose(1, 2),
                    is_discharging.transpose(1, 2),
                )

                # Compute loss
                losses = self.loss_fn(features, targets, outputs_dict, ed_solution)
                for k, v in losses.items():
                    running_loss[k] += v.item()
            return running_loss