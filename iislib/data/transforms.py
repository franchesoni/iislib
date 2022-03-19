import albumentations as A
import numpy as np
import torch


class RandomCrop:
    def __init__(self, out_size=None):
        self.aug_fn = A.CropNonEmptyMaskIfExists(*out_size)
        # if image is small we take the min of current shape to the max of
        # target shape so we are sure there is always a crop to be made
        self.scale = A.SmallestMaxSize(max_size=max(out_size), interpolation=0)
        self.out_size = out_size

    def __call__(self, image, mask):
        if (
            mask.shape[0] < self.out_size[0]
            or mask.shape[1] < self.out_size[1]
        ):  # scale up image if too small
            tsample = self.scale(image=image, mask=mask)
            image, mask = tsample["image"], tsample["mask"]
        tsample = self.aug_fn(
            image=image, mask=mask
        )  # albumentations returns a transformed object
        return tsample["image"], tsample["mask"][:, :, None]  # add one dim


class Dummy:
    def __init__(self, out_size=None):
        assert out_size is None

    def __call__(self, image, mask):
        return image, mask


def to_np(img, to_01=True):
    if 3 <= len(img.shape):  # if has more than 3-d, manually squeeze
        out = img
        while 3 < len(out.shape):
            out = out[0]
    else:
        out = img[None]  # if has 2-d, expand
    out = (
        out.permute(1, 2, 0) if out.shape[0] < 5 else out
    )  # assume imgs bigger than H=W=5, swap channels if first dim is small
    out = np.array(out)  # convert to array
    if to_01:  # take to [0, 1] assuming [0, 255] if 1 < max(img)
        out = out / 255 if 1 < out.max() else out
    else:  # assume [0, 255] and cast if 1 < max(img)
        out = out.astype(np.uint8) if 1 < out.max() else out
    return out


def norm_fn(x):
    return (x - x.min()) / (x.max() - x.min())


def scin(img):
    """Swap Channels If Needed (scin)"""
    assert img.ndim == 3, "An image has 3 dimensions! (C, H, W) or (H, W, C)"
    if isinstance(img, torch.Tensor) and min(img.shape) == img.shape[0]:
        return img.permute(2, 1, 0)
    if min(img.shape) == img.shape[-1]:
        return img
    raise RuntimeError(f"Image has shape {img.shape} which is strange")
