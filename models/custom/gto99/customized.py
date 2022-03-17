import cv2
import numpy as np
import torch

from models.custom.gto99.interaction import remove_non_fg_connected
from models.custom.gto99.networks.transforms import (
    trimap_transform,
    groupnorm_normalise_image,
)
from models.custom.gto99.networks.models import build_model
from data.transforms import to_np


def scale_input(x: np.ndarray, scale_type) -> np.ndarray:
    """Scales so that min side length is 352 and sides are divisible by 8"""
    h, w = x.shape[:2]
    h1 = int(np.ceil(h / 32) * 32)
    w1 = int(np.ceil(w / 32) * 32)
    x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
    return x_scale


def np_to_torch(x):
    return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()


def pred(
    image_np: np.ndarray, trimap_np: np.ndarray, alpha_old_np: np.ndarray, model
) -> np.ndarray:
    """Predict segmentation
    Parameters:
    image_np -- the image in rgb format between 0 and 1. Dimensions: (h, w, 3)
    trimap_np -- two channel trimap/Click map, first background then foreground. Dimensions: (h, w, 2)
    Returns:
    alpha: alpha matte/non-binary segmentation image between 0 and 1. Dimensions: (h, w)
    """
    # return trimap_np[:,:,1] + (1-np.sum(trimap_np,-1))/2
    alpha_old_np = remove_non_fg_connected(alpha_old_np, trimap_np[:, :, 1])

    h, w = trimap_np.shape[:2]
    image_scale_np = scale_input(image_np, cv2.INTER_LANCZOS4)
    trimap_scale_np = scale_input(trimap_np, cv2.INTER_NEAREST)
    alpha_old_scale_np = scale_input(alpha_old_np, cv2.INTER_LANCZOS4)

    with torch.no_grad():

        image_torch = np_to_torch(image_scale_np)
        trimap_torch = np_to_torch(trimap_scale_np)
        alpha_old_torch = np_to_torch(alpha_old_scale_np[:, :, None])

        trimap_transformed_torch = np_to_torch(trimap_transform(trimap_scale_np))
        image_transformed_torch = groupnorm_normalise_image(
            image_torch.clone(), format="nchw"
        )

        alpha = model(
            image_transformed_torch,
            trimap_transformed_torch,
            alpha_old_torch,
            trimap_torch,
        )
        alpha = cv2.resize(
            alpha[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4
        )
    alpha[trimap_np[:, :, 0] == 1] = 0
    alpha[trimap_np[:, :, 1] == 1] = 1

    alpha = remove_non_fg_connected(alpha, trimap_np[:, :, 1])
    return alpha


class args:
    use_mask_input = True
    use_usr_encoder = True
    weights = (
        "/home/franchesoni/iis/iis_framework/models/custom/gto99/InterSegSynthFT.pth"
    )
    iou_lim = None
    dataset_dir = "/home/franchesoni/adisk/iis_datasets/datasets/GrabCut/"
    predictions_dir = ""
    num_clicks = 20

model = build_model(args)
model.eval()

def initialize_z(image, target):
    z = {
        'prev_output': torch.zeros_like(target),  # (B, 1, H, W)
        # # do not use trimap for now, although slightly more inefficient
        # 'trimap': torch.zeros_like(target[0].repeat(2, 1, 1)),  # (2, H, W)
    }
    return z

def initialize_y(image, target):
    return torch.zeros_like(target)  # (B, 1, H, W)

def gto99(x, z, pcs, ncs, model=model):
    assert len(x.shape) == 4 and x.shape[0] == 1, 'Only batches of size 1 are allowed for this method'
    image = x.squeeze()  # (3, H, W)
    alpha = np.array(z['prev_output'].squeeze())  # (H, W)
    # regenerate trimap at each interaction (although we could do it iteratively)
    trimap = np.zeros((alpha.shape[0], alpha.shape[1], 2))  # (H, W, 2)
    for ncs_at_step in ncs:
        for nc in ncs_at_step[0]:  # assume batch size = 1
            if nc:  # if some negative click to do
                trimap[nc[0], nc[1], 0] = 1  
    for pcs_at_step in pcs:
        for pc in pcs_at_step[0]:
            if pc:
                trimap[pc[0], pc[1], 1] = 1
    # compute output
    image = to_np(image, to_01=True)  # (H, W, 3)
    alpha = torch.Tensor(pred(image, trimap, alpha, model))[None, None, ...]  # (1, 1, H, W)
    y, z = alpha, {'prev_output': alpha}
    return y, z

