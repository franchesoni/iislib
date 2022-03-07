import os
import time
from argparse import ArgumentParser
import json

import pytorch_lightning as pl


def get_dataloaders():
    from data.datasets.coco_lvis import CocoLvisDataset
    from data.transformations import RandomCrop
    from data.region_selector import random_single
    from data.iis_dataset import RegionDataset, RegionDataLoader

    # train data
    seg_dataset = CocoLvisDataset(
        "/home/franchesoni/adisk/iis_datasets/datasets/LVIS", split="train"
    )
    region_selector = random_single
    augmentator = RandomCrop(out_size=(224, 224))
    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    train_iis_dataloader = RegionDataLoader(iis_dataset, batch_size=2, num_workers=0)

    # val data
    val_seg_dataset = CocoLvisDataset(
        "/home/franchesoni/adisk/iis_datasets/datasets/LVIS", split="val"
    )
    val_region_selector = random_single
    val_augmentator = RandomCrop(out_size=(224, 224))
    val_iis_dataset = RegionDataset(
        val_seg_dataset, val_region_selector, val_augmentator
    )
    val_iis_dataloader = RegionDataLoader(val_iis_dataset, batch_size=2, num_workers=0)
    return train_iis_dataloader, val_iis_dataloader


def get_model():
    import segmentation_models_pytorch as smp
    from models.lightning import LitIIS
    from models.iis_smp_wrapper import EarlySMP
    from engine.metrics import mse

    lit_model = LitIIS(
        mse,
        EarlySMP,
        iis_model_args_list=[smp.Unet, {"encoder_name": "mobilenet_v2", "encoder_weights": "imagenet"}],
        iis_model_kwargs_dict={"in_channels": 6},
    )
    return lit_model


if __name__ == "__main__":
    train_dataloader, val_dataloader = get_dataloaders()
    model = get_model()

    trainer = pl.Trainer(gpus=1, precision=16)
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


