import functools
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms
from clicking.robots import robot_01
from clicking.robots import robot_02
from clicking.robots import robot_03
from data.iis_dataset import EvaluationDataset
from data.transforms import norm_fn
from engine.metrics import eval_metrics
from torchvision.transforms.functional import center_crop
from torchvision.transforms.functional import resize

sys.path.append("/home/franchesoni/iis/iislib/tests/")
from visualization import visualize_on_test


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


def get_model(version="last"):
    from models.lightning import LitIIS

    results_dir = "/home/franchesoni/iis/iislib/results"
    logs_dir = Path(results_dir) / "lightning_logs/"
    versions = [vname for vname in os.listdir(logs_dir)]
    if version == "last":
        version = max(int(vname.split("_")[-1]) for vname in versions)
    checkpoint_path = glob.glob(
        str(logs_dir / f"version_{version}" / "checkpoints") + "/*"
    )[0]
    model = LitIIS.load_from_checkpoint(checkpoint_path, robot_click=None)
    return model.model, model.model.init_z, model.model.init_y
    # encoding_fn = encode_disks

    # def fwd_lit_iis(image, z, pcs, ncs):
    #     pos_encoding, neg_encoding = z["pos_encoding"], z["neg_encoding"]
    #     pos_encoding, neg_encoding = encode_clicks(
    #         pcs, ncs, encoding_fn, pos_encoding, neg_encoding
    #     )
    #     x, aux = image, torch.cat(
    #         (pos_encoding, neg_encoding, z["prev_output"]), dim=1
    #     )
    #     prev_output = torch.sigmoid(model(x, aux))
    #     return prev_output, {
    #         "prev_output": prev_output,
    #         "pos_encoding": pos_encoding,
    #         "neg_encoding": neg_encoding,
    #     }

    # def initialize_z(image, target):
    #     prev_output = torch.zeros_like(image[:, :1])
    #     pos_encoding, neg_encoding = [prev_output.clone()] * 2
    #     return {
    #         "prev_output": prev_output,
    #         "pos_encoding": pos_encoding,
    #         "neg_encoding": neg_encoding,
    #     }

    # def initialize_y(image, target):
    #     return torch.zeros_like(image[:, :1])

    # return fwd_lit_iis, initialize_z, initialize_y


def get_model_gto99():
    from models.custom.gto99.customized import (
        gto99,
        initialize_y,
        initialize_z,
    )

    return gto99, initialize_z, initialize_y


def get_model_ritm():
    from models.custom.ritm.customized import initialize_y, initialize_z, ritm

    return ritm, initialize_z, initialize_y


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
        new_shape = [
            min(img.shape[-2:])
        ] * 2  # (224, 224)  #tuple(32*(side//32) for side in img.shape[-2:])
    img, mask = center_crop(img, new_shape), center_crop(mask, new_shape)
    img, mask = resize(
        img, (224, 224), torchvision.transforms.InterpolationMode.NEAREST
    ), resize(
        mask, (224, 224), torchvision.transforms.InterpolationMode.NEAREST
    )
    return img, mask


def get_dataset():
    from data.datasets.berkeley import BerkeleyDataset

    seg_dataset = BerkeleyDataset(
        "/home/franchesoni/adisk/iis_datasets/datasets/Berkeley"
    )
    ds = EvaluationDataset(seg_dataset, augmentator=augmentator)
    return ds


def test():
    exp_prefix = "mar24_"
    # model_names = ['gto99', 'ritm', 'ours']
    model_names = ["ours_early"]
    robots = [robot_01, robot_02, robot_03]
    robot_prefix = ["r01_", "r02_", "r03_"]

    seed = 0
    max_n_clicks = 20
    destdir = "/home/franchesoni/iis/iislib/results/benchmark"
    compute_scores = functools.partial(
        eval_metrics,
        num_classes=1,
        ignore_index=0,
        metrics=["mIoU", "mDice", "mFscore"],
    )

    for model_name in model_names:
        for robot_ind, robot in enumerate(robots):
            prefix = exp_prefix + robot_prefix[robot_ind]

            if model_name == "ritm":
                model, init_z, init_y = get_model_ritm()
            elif model_name == "gto99":
                model, init_z, init_y = get_model_gto99()
            elif model_name == "ours":
                model, init_z, init_y = get_model()
            elif model_name == "ours_early":
                model, init_z, init_y = get_model("35")
            ds = get_dataset()
            pl.seed_everything(seed)
            dl = torch.utils.data.DataLoader(ds, batch_size=1)
            scores = []

            for bi, batch in enumerate(dl):
                image, target = (
                    batch["image"].float(),
                    batch["mask"].float(),
                )  # (B, C, H, W)

                z = init_z(image, target)
                y = init_y(image, target)  # (B, 1, H, W)
                pcs, ncs = [], []  # two click_seq
                scores.append([])

                for iter_ind in range(max_n_clicks):
                    pcs, ncs = robot(y, target, n_points=1, pcs=pcs, ncs=ncs)
                    y, z = model(image, z, pcs, ncs)  # (B, 1, H, W), z
                    if bi < 3:
                        visualize_on_test(
                            to_np(image[0]),
                            np.array(target[0][0]),
                            output=np.array(y[0][0].detach()),
                            pcs=pcs,
                            ncs=ncs,
                            name=f"{prefix}{model_name}_{str(bi).zfill(2)}_{str(iter_ind).zfill(2)}",
                            destdir=destdir,
                        )

                    ss = compute_scores(
                        1 * (0.5 < norm_fn(y)),
                        norm_fn(target),
                    )
                    scores[-1].append(ss)
                    print(f"done with batch {bi} click {iter_ind}, score={ss}")

            np.save(
                os.path.join(destdir, f"scores_{prefix}{model_name}.npy"),
                scores,
            )


if __name__ == "__main__":
    test()
