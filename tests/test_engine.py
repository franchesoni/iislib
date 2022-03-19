def test_training_logic():
    import pytorch_lightning as pl
    import segmentation_models_pytorch as smp
    import torch
    from data.datasets.coco_lvis import CocoLvisDataset
    from data.iis_dataset import RegionDataLoader, RegionDataset, visualize
    from data.region_selector import random_single
    from data.transforms import RandomCrop
    from engine.training_logic import interact
    from models.wrappers.iis_smp_wrapper import EarlySMP

    pl.seed_everything(0)

    # data
    seg_dataset = CocoLvisDataset(
        "/home/franchesoni/adisk/iis_datasets/datasets/LVIS"
    )
    region_selector = random_single
    augmentator = RandomCrop(out_size=(224, 224))
    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    iis_dataloader = torch.utils.data.DataLoader(
        iis_dataset, batch_size=2, num_workers=0
    )
    # model
    model = EarlySMP(
        smp.Unet,
        {"encoder_name": "mobilenet_v2", "encoder_weights": "imagenet"},
        in_channels=6,
    )

    for ind, batch in enumerate(iis_dataloader):
        print(ind)
        images, masks, infos = batch["image"], batch["mask"], batch["info"]
        output, pc_mask, nc_mask, pcs, ncs = interact(
            model, batch, interaction_steps=3, clicks_per_step=3
        )
        visualize(images[0].permute(1, 2, 0), "image")
        visualize(masks[0][0], "mask")
        visualize(output[0][0].detach().numpy(), "output")
        visualize(pc_mask[0][0], "pcs")
        visualize(nc_mask[0][0], "ncs")
        breakpoint()
