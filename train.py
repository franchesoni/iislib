import torch
import pytorch_lightning as pl


def get_dataloaders(num_workers=12, batch_size=256):
    from data.datasets.coco_lvis import CocoLvisDataset
    from data.transforms import RandomCrop
    from data.region_selector import random_single
    from data.iis_dataset import RegionDataset, RegionDataLoader

    # train data
    seg_dataset = CocoLvisDataset(
        "/home/franchesoni/adisk/iis_datasets/datasets/LVIS", split="train"
    )
    region_selector = random_single
    augmentator = RandomCrop(out_size=(224, 224))
    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    train_iis_dataloader = torch.utils.data.DataLoader(
        iis_dataset, batch_size=batch_size, num_workers=num_workers
    )

    # val data
    val_seg_dataset = CocoLvisDataset(
        "/home/franchesoni/adisk/iis_datasets/datasets/LVIS", split="val"
    )
    val_region_selector = random_single
    val_augmentator = RandomCrop(out_size=(224, 224))
    val_iis_dataset = RegionDataset(
        val_seg_dataset, val_region_selector, val_augmentator
    )
    val_iis_dataloader = torch.utils.data.DataLoader(
        val_iis_dataset, batch_size=batch_size, num_workers=num_workers
    )
    return train_iis_dataloader, val_iis_dataloader


def get_model(num_workers=4, batch_size=8, hacky=False):
    import segmentation_models_pytorch as smp
    from models.lightning import LitIIS
    from models.wrappers.iis_smp_wrapper import EarlySMP
    from engine.metrics import mse

    lit_model = LitIIS(
        mse,
        EarlySMP,
        iis_model_args_list=[
            smp.Unet,
            {"encoder_name": "mobilenet_v2", "encoder_weights": "imagenet"},
        ],
        iis_model_kwargs_dict={"in_channels": 6},
    )
    if hacky:
        # hacky way to be able to tune the lr and batch size
        LitIIS.train_dataloader = lambda self: get_dataloaders(num_workers=self.num_workers, batch_size=self.batch_size)[0]
        LitIIS.val_dataloader = lambda self: get_dataloaders(num_workers=self.num_workers, batch_size=self.batch_size)[1]
        lit_model.num_workers = num_workers
        lit_model.batch_size = batch_size
    return lit_model


if __name__ == "__main__":
    seed = 0
    tune = False
    num_workers, batch_size = 6, 24

    pl.seed_everything(seed)
    if tune:
        model = get_model(num_workers=num_workers, batch_size=batch_size, hacky=True)
    else:
        train_dataloader, val_dataloader = get_dataloaders(num_workers=num_workers, batch_size=batch_size)
        model = get_model(num_workers=num_workers, batch_size=batch_size, hacky=False)

    trainer = pl.Trainer(
        # training options
        gpus=1,
        precision=16,
        max_epochs=1000,

        # my options
        log_every_n_steps=1,
        deterministic=True,
        benchmark=False,

        # optimization options
        auto_scale_batch_size='binsearch',
        auto_lr_find=True,

        # debugging options:
        callbacks=[pl.callbacks.DeviceStatsMonitor()],
        profiler='simple',
        # profiler=pl.profiler.AdvancedProfiler(filename='profile_report.txt'),
        # fast_dev_run=True,
        # overfit_batches=10,
    )

    if tune:
        trainer.tune(model)
        trainer.fit(model)

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
