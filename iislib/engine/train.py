import functools
import os  # optional: for extracting basename / creating new filepath
import shutil
import time

import pytorch_lightning as pl
import torch


def get_dataloaders(num_workers=12, batch_size=256):
    from data.datasets.sbd import SBDDataset
    from data.iis_dataset import RegionDataset
    from data.region_selector import random_single
    from data.transforms import (
        RandomCrop,
        composed_func,
        fix_mask_shape,
        to_channel_first,
        to_channel_last,
    )

    # train data
    seg_dataset = SBDDataset(
        "/home/franchesoni/adisk/iis_datasets/datasets/SBD", split="train"
    )
    region_selector = random_single
    augmentator = composed_func(
        [
            fix_mask_shape,
            to_channel_last,
            RandomCrop(out_size=(224, 224)),
            to_channel_first,
        ]
    )
    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    train_iis_dataloader = torch.utils.data.DataLoader(
        iis_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    # val data
    val_seg_dataset = SBDDataset(
        "/home/franchesoni/adisk/iis_datasets/datasets/SBD", split="test"
    )
    val_region_selector = random_single
    val_augmentator = augmentator
    val_iis_dataset = RegionDataset(
        val_seg_dataset, val_region_selector, val_augmentator
    )
    val_iis_dataloader = torch.utils.data.DataLoader(
        val_iis_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_iis_dataloader, val_iis_dataloader


def get_model(num_workers=4, batch_size=8, hacky=False):
    import segmentation_models_pytorch as smp
    from engine.metrics import mse
    from models.lightning import LitIIS
    from models.wrappers.iis_smp_wrapper import EarlySMP, EncodeSMP
    from clicking.encode import encode_disks_from_scratch
    from clicking.robots import build_robot_mix, robot_01, robot_02, robot_03
    from engine.metrics import eval_metrics

    lit_model = LitIIS(
        mse,
        build_robot_mix([robot_01, robot_02, robot_03], [3, 6, 1]),
        # EarlySMP,
        EncodeSMP,
        iis_model_kwargs_dict={
            "smp_model_class": smp.Unet,
            "smp_model_kwargs_dict": {
                # "encoder_name": "dpn68b",
                # "encoder_weights": "imagenet+5k",
                "encoder_name": "mobilenet_v2",
                "encoder_weights": "imagenet",
            },
            "click_encoder": encode_disks_from_scratch,
        },
        training_metrics=functools.partial(
            eval_metrics,
            num_classes=1,
            ignore_index=0,
            metrics=["mIoU", "mDice", "mFscore"],
        ),
        max_interactions=4,
    )
    if hacky:
        # hacky way to be able to tune the lr and batch size
        LitIIS.train_dataloader = lambda self: get_dataloaders(
            num_workers=self.num_workers, batch_size=self.batch_size
        )[0]
        LitIIS.val_dataloader = lambda self: get_dataloaders(
            num_workers=self.num_workers, batch_size=self.batch_size
        )[1]
        lit_model.num_workers = num_workers
        lit_model.batch_size = batch_size
    return lit_model


if __name__ == "__main__":
    st = time.time()
    seed = 0
    tune = False
    # num_workers, batch_size = 6, 128
    num_workers, batch_size = 4, 16

    pl.seed_everything(seed)
    if tune:
        model = get_model(
            num_workers=num_workers, batch_size=batch_size, hacky=True
        )
    else:
        train_dataloader, val_dataloader = get_dataloaders(
            num_workers=num_workers, batch_size=batch_size
        )
        model = get_model(
            num_workers=num_workers, batch_size=batch_size, hacky=False
        )

    class SaveScriptCallback(pl.callbacks.Callback):
        def on_train_start(self, trainer, pl_module):
            if log_dir := trainer.log_dir:
                copied_script_name = (
                    time.strftime("%Y-%m-%d_%H%M")
                    + "_"
                    + os.path.basename(__file__)
                )
                shutil.copy(
                    __file__, os.path.join(log_dir, copied_script_name)
                )

    trainer = pl.Trainer(
        # training options
        default_root_dir="../results",
        gpus=1,
        precision=16,
        # max_epochs=1000,  # if commented, run forever
        # my options
        log_every_n_steps=1,
        deterministic=True,
        benchmark=False,
        # # optimization options
        # auto_scale_batch_size="binsearch",
        # auto_lr_find=True,
        # debugging options:
        callbacks=[
            SaveScriptCallback()
        ],  # pl.callbacks.DeviceStatsMonitor()],
        profiler="simple",
        # profiler=pl.profiler.AdvancedProfiler(filename='profile_report.txt'),
        # fast_dev_run=True,
        # overfit_batches=1,
    )
    print("Finished setup")
    print(f"Set up time: {time.time() - st}")
    print("=" * 40)
    print("Start training")

    if tune:
        trainer.tune(model)
        trainer.fit(model)
    else:
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
