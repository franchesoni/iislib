import sys
from typing import Callable
from typing import Union

import pytorch_lightning as pl
import torch
import torchvision
from engine.training_logic import interact

sys.path.append("../tests")
from visualization import visualize


class LitIIS(pl.LightningModule):
    def __init__(
        self,
        loss_fn,
        robot_click,
        init_robot_click,
        iis_model_cls,
        iis_model_args_list=None,
        iis_model_kwargs_dict=None,
        training_metrics=None,
        validation_metrics=None,
        interaction_steps=None,
        max_interactions=4,
        max_init_clicks=5,
        lr=0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        # define generic model
        iis_model_args_list = iis_model_args_list or []
        iis_model_kwargs_dict = iis_model_kwargs_dict or {}
        self.model = iis_model_cls(
            *iis_model_args_list, **iis_model_kwargs_dict
        )
        self.loss_fn = loss_fn
        self.robot_click = robot_click
        self.init_robot_click = init_robot_click
        self.training_metrics = training_metrics or {}
        self.validation_metrics = validation_metrics or training_metrics
        self.interaction_steps = interaction_steps
        self.max_interactions = max_interactions
        self.max_init_clicks = max_init_clicks
        self.lr = lr

    def forward(self, x, z, pcs, ncs):
        return self.model(x, z, pcs, ncs)

    def log_metrics(
        self,
        y: torch.Tensor,
        target: torch.Tensor,
        metrics: Union[Callable, dict],
        prefix: str = "",
        name: str = "",
    ):
        """Logs metrics computed over `y` and `target` with flexibility.

        Args:
            y (torch.Tensor): predicted mask, (B, 1, H, W)
            target (torch.Tensor): target, (B, 1, H, W)
            metrics (Union[Callable, dict]):
                one of:
                - dict of key:values to log
                - dict of functions, key:function(y, target) is logged
                - function that returns a dict and logs its key:values
            prefix (str, optional): something to append to the name. Defaults to ''.
            name (str, optional): something to prepend to the metric name. Defaults to ''.

        Raises:
            ValueError: if `metrics` is not on the scenarios above
        """
        if isinstance(metrics, dict):
            for metric_name in metrics:
                if callable(metric := metrics[metric_name]):
                    # if its a function, compute and log the results
                    self.log_metrics(
                        y, target, metric, prefix=prefix, name=metric_name
                    )
                else:  # assume a value and log it
                    self.log(
                        "_".join([prefix, name, metric_name]), float(metric)
                    )
        elif callable(metrics):
            if isinstance(computed_metrics := metrics(y, target), dict):
                # assume a dict of results
                self.log_metrics(
                    None, None, computed_metrics, prefix=prefix, name=name
                )
            else:  # just a value
                self.log("_".join([prefix, name]), float(computed_metrics))
        elif metrics is not None:  # metrics=None ignores everyghin
            raise ValueError("`metrics` should be a function or a dict")

    def training_step(self, batch, batch_idx):
        y, z, pcs, ncs = interact(
            self.model,
            self.model.init_z,
            self.model.init_y,
            self.robot_click,
            self.init_robot_click,
            batch,
            interaction_steps=self.interaction_steps,
            max_interactions=self.max_interactions,
            clicks_per_step=1,
            max_init_clicks=self.max_init_clicks,
            batch_idx=batch_idx,
        )
        target = batch["mask"]
        loss = torch.mean(self.loss_fn(y, target))
        self.log("train_loss", loss)  # lightning
        self.log_metrics(
            y.detach(), target.detach(), self.training_metrics, prefix="train"
        )  # custom
        return loss

    def validation_step(self, batch, batch_idx):
        y, z, pcs, ncs = interact(
            self.model,
            self.model.init_z,
            self.model.init_y,
            self.robot_click,
            self.init_robot_click,
            batch,
            interaction_steps=self.interaction_steps or self.max_interactions,
            max_interactions=None,  # no randomness involved when validating
            clicks_per_step=1,
            max_init_clicks=self.max_init_clicks,
            batch_idx=batch_idx,
        )
        target = batch["mask"]
        loss = self.loss_fn(y, target)
        self.log("val_loss", loss)  # lightning
        self.log_metrics(
            y, target, self.training_metrics, prefix="val"
        )  # custom
        # save images
        grid1 = torchvision.utils.make_grid([batch["image"][0]])
        grid2 = torchvision.utils.make_grid([batch["mask"][0], y[0]])
        self.logger.experiment.add_image(
            "validation input", grid1, self.global_step
        )
        self.logger.experiment.add_image(
            "validation result", grid2, self.global_step
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=1e-4
        )
