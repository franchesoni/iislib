import torch

def test_data__iis_dataset():
    from data.datasets.coco_lvis import CocoLvisDataset
    from data.transforms import RandomCrop
    from data.region_selector import random_single
    from data.iis_dataset import RegionDataset, visualize

    import pytorch_lightning as pl

    pl.seed_everything(0)

    seg_dataset = CocoLvisDataset("/home/franchesoni/adisk/iis_datasets/datasets/LVIS")
    region_selector = random_single
    augmentator = RandomCrop(out_size=(224, 224))
    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    iis_dataloader = torch.utils.data.DataLoader(iis_dataset, batch_size=2, num_workers=0)
    for ind, batch in enumerate(iis_dataloader):
        print(ind)
        images, masks, infos = (batch['image'], batch['mask'], batch['info'])
        visualize(images[0], "image")
        visualize(masks[0], "mask")
        breakpoint()


def test_data__datasets__coco_lvis():
    import matplotlib.pyplot as plt
    from data.datasets.coco_lvis import CocoLvisDataset

    dataset = CocoLvisDataset(
        "/home/franchesoni/adisk/iis_datasets/datasets/LVIS"
    )
    for ind in range(100):
        sample = dataset.get_sample(ind)
        assert (
            dataset.get_sample(ind)[0] == sample[0]
        ).all()  # check that it works the same always
        image, layers, info = sample
        plt.imshow(image)
        plt.show()
        plt.savefig("image.png")
        for idx, layer in enumerate(np.moveaxis(layers, 2, 0)):
            plt.imshow(layer)
            plt.show()
            plt.savefig(f"layer_{idx}.png")
        print(info)
        breakpoint()

def test_data__clicking():
    from data.datasets.coco_lvis import CocoLvisDataset
    from data.transforms import RandomCrop
    from data.region_selector import random_single
    from data.iis_dataset import RegionDataset, visualize, scin
    from data.clicking import get_positive_clicks_batch, visualize_clicks

    import pytorch_lightning as pl

    pl.seed_everything(0)

    iis_dataloader = torch.data.utils.DataLoader(
        RegionDataset(
            seg_dataset=CocoLvisDataset(
                "/home/franchesoni/adisk/iis_datasets/datasets/LVIS"
            ),
            region_selector=random_single,
            augmentator=RandomCrop(out_size=(224, 224)),
        ),
        batch_size=2,
        num_workers=0,
    )

    for ind, batch in enumerate(iis_dataloader):
        print(f"index {ind}")
        ncs = []
        images, masks, infos = batch
        masks = np.array(masks.squeeze())
        pcs = get_positive_clicks_batch(
            10, masks[:1], near_border=True, uniform_probs=True, erode_iters=5
        )
        visualize_clicks(images[0], masks[:1], 0.3, pcs, ncs, "vis1")
        pcs = get_positive_clicks_batch(
            100, masks[:1], near_border=False, uniform_probs=True, erode_iters=15
        )
        visualize_clicks(images[0], masks[:1], 0.3, pcs, ncs, "vis2")
        pcs = get_positive_clicks_batch(
            100, masks[:1], near_border=False, uniform_probs=False, erode_iters=15
        )
        visualize_clicks(images[0], masks[:1], 0.3, pcs, ncs, "vis3")
        pcs = get_positive_clicks_batch(
            100, masks[:1], near_border=False, uniform_probs=True, erode_iters=0
        )
        visualize_clicks(images[0], masks[:1], 0.3, pcs, ncs, "vis4")
        breakpoint()

def test_datasets__coco_lvis__drop_from_first_if_not_in_second():
    from data.datasets.coco_lvis import drop_from_first_if_not_in_second
    alist = list(range(10))  # always the same

    blist = [0, 1, 3, 4, 7, 9]
    expected = [a for a in alist if a in blist]
    output, _ = drop_from_first_if_not_in_second(alist, blist)
    assert  output == expected, f'{alist}, {blist}, {expected}, {output}'

    blist = [0, 1, 3, 4, 7]
    expected = [a for a in alist if a in blist]
    output, _ = drop_from_first_if_not_in_second(alist, blist)
    assert  output == expected, f'{alist}, {blist}, {expected}, {output}'

    blist = [1, 3, 4, 7, 9]
    expected = [a for a in alist if a in blist]
    output, _ = drop_from_first_if_not_in_second(alist, blist)
    assert  output == expected, f'{alist}, {blist}, {expected}, {output}'

    blist = list(range(-3, 5))
    expected = [a for a in alist if a in blist]
    output, _ = drop_from_first_if_not_in_second(alist, blist)
    assert  output == expected, f'{alist}, {blist}, {expected}, {output}'

    blist = list(range(6, 13))
    expected = [a for a in alist if a in blist]
    output, _ = drop_from_first_if_not_in_second(alist, blist)
    assert  output == expected, f'{alist}, {blist}, {expected}, {output}'

