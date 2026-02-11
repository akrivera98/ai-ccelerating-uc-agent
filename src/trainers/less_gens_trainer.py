import torch
from tqdm import tqdm
from src.trainers.base_trainer import BaseTrainer
from src.registry import registry


@registry.register_trainer("less_gens_scipy_trainer")
class ScpiyLpTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.ed_model_name = self.config.ed_model.name
        self.ed_model_params = self.config.ed_model.params

    def load_model(self):
        model_name = self.config.model.name
        model_params = self.config.model.params.to_dict()

        # ---- inject indices computed by dataset ----
        model_params["G"] = len(self.dataset.gen_names_all)
        model_params["pred_gen_idx"] = self.dataset.pred_idx.tolist()
        model_params["fixed_on_idx"] = self.dataset.fixed_on_idx.tolist()
        model_params["fixed_off_idx"] = self.dataset.fixed_off_idx.tolist()

        model_class = registry.get_model(model_name)
        self.model = model_class(**model_params).to(self.device)

    def load_ed_model(self):
        ed_model_name = getattr(self.config.ed_model, "name", None)
        if ed_model_name is None:
            self.ed_model = None
            return

        ed_params = self.config.ed_model.params.to_dict()

        # inject LP generator subset (omit fixed-off)
        ed_params["G_full"] = len(self.dataset.gen_names_all)
        ed_params["lp_gen_idx"] = self.dataset.lp_gen_idx.tolist()  # pred and fixed_on

        ed_model_class = registry.get_ed_model(ed_model_name)
        self.ed_model = ed_model_class(**ed_params)

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

            # Solve ED problem with SciPy LP solver
            ed_obj = self.ed_model.objective(
                load,
                solar_max,
                wind_max,
                is_on,
                is_charging,
                is_discharging,
            )

            # Compute loss
            losses = self.loss_fn(features, targets, outputs_dict, ed_obj)

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

                # Solve ED problem with SciPy LP solver
                ed_obj = self.ed_model.objective(
                    load,
                    solar_max,
                    wind_max,
                    is_on,
                    is_charging,
                    is_discharging,
                )

                # Compute loss
                losses = self.loss_fn(features, targets, outputs_dict, ed_obj)
                for k, v in losses.items():
                    running_loss[k] += v.item()
            return running_loss
