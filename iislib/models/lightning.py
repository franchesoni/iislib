import pytorch_lightning as pl
import torch
from engine.training_logic import interact


class LitIIS(pl.LightningModule):
    def __init__(
        self,
        loss_fn,
        robot_click,
        iis_model_cls,
        iis_model_args_list=[],
        iis_model_kwargs_dict={},
        training_metrics={},
        validation_metrics={},
        interaction_steps=3,
        lr=0.0001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = iis_model_cls(
            *iis_model_args_list, **iis_model_kwargs_dict
        )
        self.loss_fn = loss_fn
        self.robot_click = robot_click
        self.training_metrics = training_metrics
        self.validation_metrics = (
            validation_metrics if len(validation_metrics) else training_metrics
        )
        self.interaction_steps = interaction_steps
        self.lr = lr

    def forward(self, x, z, pcs, ncs):
        return self.model(x, z, pcs, ncs)

    def training_step(self, batch, batch_idx):
        y, z, pcs, ncs = interact(
            self.model,
            self.init_z,
            self.init_y,
            self.robot_click,
            batch,
            self.interaction_steps,
            max_interactions=None,
            clicks_per_step=2,
            batch_idx=batch_idx,
        )
        target = batch["mask"]
        breakpoint()
        loss = self.loss_fn(
            y.squeeze(), target.squeeze()
        )  # get dimensions right
        self.log("train_loss", loss)
        for metric_name in self.training_metrics:
            self.log(
                metric_name, self.training_metrics[metric_name](output, target)
            )
        return loss

    def validation_step(self, batch, batch_idx):
        y, z, pcs, ncs = interact(
            self.model,
            self.init_z,
            self.init_y,
            self.robot_click,
            batch,
            self.interaction_steps,
            max_interactions=None,
            clicks_per_step=2,
            batch_idx=batch_idx,
        )
        target = batch["mask"]
        breakpoint()
        loss = self.loss_fn(
            y.squeeze(), target.squeeze()
        )  # get dimensions right
        self.log("val_loss", loss)
        for metric_name in self.validation_metrics:
            self.log(
                metric_name,
                self.validation_metrics[metric_name](output, target),
            )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
