import os
import glob
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import center_crop, resize
import torchvision.transforms
import pytorch_lightning as pl

from clicking.robots import robot_01
from clicking.encode import encode_clicks, encode_disks
from data.iis_dataset import EvaluationDataset
from engine.metrics import eval_metrics



"""Testing of any method
- Loads batches from an EvaluationDataset
- Runs the IIS with some `robot` and saving results
"""


def to_np(img):
    if 3 <= len(img.shape):
        out = img

        while 3 < len(out.shape):
            out = out[0]
    else:
        out = img[None]
    out = (
        out.permute(1, 2, 0) if out.shape[0] < 4 else out
    )  # assume images bigger than 5x5
    out = np.array(out)
    out = out / 255 if 1 < out.max() else out
    return out


def get_model():
    from models.lightning import LitIIS

    logs_dir = Path(__file__).parent / "lightning_logs/"
    versions = [vname for vname in os.listdir(logs_dir)]
    last_version = max([int(vname.split("_")[-1]) for vname in versions])
    checkpoint_path = glob.glob(
        str(logs_dir / f"version_{last_version}" / "checkpoints") + "/*"
    )[0]
    model = LitIIS.load_from_checkpoint(checkpoint_path)
    return model

def get_sample(dataset, sample_ind):
    img, mask = (
        torch.Tensor(dataset[sample_ind][0]).permute(2, 0, 1),
        torch.Tensor(dataset[sample_ind][1]).squeeze()[None],
    )
    assert img.shape[-2:] == mask.shape[-2:]
    assert len(img.shape) == len(mask.shape) == 3
    return augmentator(img, mask)

def augmentator(img, mask, new_shape=None):
    img, mask = (
        torch.Tensor(img).permute(2, 0, 1),
        torch.Tensor(mask).squeeze()[None],
    )
    if new_shape is None:
        new_shape = [min(img.shape[-2:])] * 2  # (224, 224)  #tuple(32*(side//32) for side in img.shape[-2:])
    img, mask = center_crop(img, new_shape), center_crop(mask, new_shape)
    img, mask = resize(img, (224, 224), torchvision.transforms.InterpolationMode.NEAREST), resize(mask, (224, 224), torchvision.transforms.InterpolationMode.NEAREST)
    return img, mask

def get_dataset():
    from data.datasets.berkeley import BerkeleyDataset
    seg_dataset = BerkeleyDataset("/home/franchesoni/adisk/iis_datasets/datasets/Berkeley")
    ds = EvaluationDataset(seg_dataset, augmentator=augmentator)
    return ds


def test():
    max_n_clicks = 4
    encoding_fn = encode_disks  
    compute_scores = eval_metrics

    robot = robot_01
    model = get_model()
    ds = get_dataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=2)
    scores = []

    with torch.no_grad():  # dont use this line in general
        for bi, batch in enumerate(dl):
            image, target = batch["image"], batch["mask"]
            prev_output = torch.zeros_like(image[:, :1])
            pos_encoding, neg_encoding = [prev_output.clone()]*2
            pcs, ncs = [], []
            scores.append([])

            for iter_ind in range(max_n_clicks):
                pcs, ncs = robot(prev_output, target, n_points=1, pcs=pcs, ncs=ncs)
                pos_encoding, neg_encoding = encode_clicks(pcs, ncs, encoding_fn, pos_encoding, neg_encoding)
                x, aux = image, torch.cat((pos_encoding, neg_encoding, prev_output), dim=1)
                prev_output = torch.sigmoid(model(x, aux))
                ss = compute_scores(1*(0.5 < prev_output), target, num_classes=1, ignore_index=0, metrics=['mIoU', 'mDice', 'mFscore'])
                scores[-1].append(ss)
                print(f'done with batch {bi} click {iter_ind}')


def try_one_image(sample_ind=0, seed=0):
    pl.seed_everything(seed)
    model = get_model()
    ds = get_dataset()
    img, mask = get_sample(ds, sample_ind)

    from data.iis_dataset import visualize
    from engine.training_logic import interact_single_step

    visualize(to_np(img), "image")
    visualize(to_np(mask), "mask")
    output, pcs, ncs, pc_mask, nc_mask = interact_single_step(
        True, model, img, mask, prev_output=None
    )
    visualize(to_np(output), "output1")
    visualize(to_np(pc_mask), "pc_mask1")
    visualize(to_np(nc_mask), "nc_mask1")
    for i in range(2, 20):
        output, pcs, ncs, pc_mask, nc_mask = interact_single_step(
            False,
            model,
            img,
            mask,
            prev_output=output,
            pc_mask=pc_mask,
            nc_mask=nc_mask,
            pcs=pcs,
            ncs=ncs,
        )
        visualize(to_np(output), f"output_{i}")
        visualize(to_np(pc_mask), f"pc_mask_{i}")
        visualize(to_np(nc_mask), f"nc_mask_{i}")

    breakpoint()

if __name__=='__main__':
    test()