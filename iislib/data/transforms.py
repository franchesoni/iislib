from typing import Union

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations import DualTransform
from albumentations.augmentations import functional as F
from albumentations.augmentations.geometric import resize


def composed_func(list_of_funcs):
    def transform(
        image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        for func in list_of_funcs:
            ret = func(image=image, mask=mask)
            if isinstance(ret, tuple):
                image, mask = ret
            elif isinstance(ret, dict):
                image, mask = ret["image"], ret["mask"]
            else:
                raise TypeError("Return type is not a tuple or dict")
        return image, mask

    return transform


def fix_mask_shape(
    image: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
) -> tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """Adds trailing dimension to mask if needed in image's channel position

    Args:
        image (Union[torch.Tensor, np.ndarray]): image, (H, W, C) or (C, H, W)
        mask (Union[torch.Tensor, np.ndarray]): mask, (H, W), or

    Raises:
        ValueError: if output mask shape is not similar to image shape
        (flexibility on the channel dim)

    Returns:
        tuple[Union[torch.Tensor, np.ndarray]: input image
        Union[torch.Tensor, np.ndarray]]: transformed mask
    """
    if len(mask.shape) == 2:
        if image.shape[0] == min(image.shape):
            mask = mask[None, ...]  # (1, H, W)
        elif image.shape[-1] == min(image.shape):
            mask = mask[..., None]  # (H, W, 1)
        else:
            raise ValueError(f"Image has shape {image.shape} which is strange")
    if not (np.argmin(mask.shape) == np.argmin(image.shape)):
        raise ValueError(
            f"output `mask` and `image` should have same channel order but \
                have {mask.shape} and {image.shape} shapes respectively"
        )
    return image, mask


def to_channel_first(
    image: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
) -> tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    return to_channel_first_single(image), to_channel_first_single(mask)


def to_channel_first_single(
    image: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Converts to (C, H, W)"""
    assert (
        image.ndim == 3
    ), f"An image should have 3 dimensions! (C, H, W) or (H, W, C). Yours has shape {image.shape}"
    # if we need to do something
    if min(image.shape) == image.shape[-1]:
        if isinstance(image, torch.Tensor):
            return image.permute(2, 0, 1)
        if isinstance(image, np.ndarray):
            return np.transpose(image, (2, 0, 1))
        raise ValueError(
            f"image type is {type(image)} which is not a Tensor nor an ndarray"
        )
    # if we don't need to change anything
    if min(image.shape) == image.shape[0]:
        return image
    raise ValueError(f"Image has shape {image.shape} which is strange")


def to_channel_last(
    image: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
) -> tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """Converts to (H, W, C)"""
    return to_channel_last_single(image), to_channel_last_single(mask)


def to_channel_last_single(
    image: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """Converts to (H, W, C)"""
    assert (
        image.ndim == 3
    ), f"An image should have 3 dimensions! (C, H, W) or (H, W, C). \
        Yours has shape {image.shape}"
    # if we need to do something
    if min(image.shape) == image.shape[0]:
        if isinstance(image, torch.Tensor):
            return image.permute(1, 2, 0)
        if isinstance(image, np.ndarray):
            return np.transpose(image, (1, 2, 0))
        raise ValueError(
            f"image type is {type(image)} which is not a Tensor nor an ndarray"
        )
    # if we don't need to change anything
    if min(image.shape) == image.shape[-1]:
        return image
    raise ValueError(f"Image has shape {image.shape} which is strange")


class RandomCrop:
    def __init__(self, out_size: tuple):
        self.aug_fn = A.CropNonEmptyMaskIfExists(*out_size)
        # if image is small we take the min of current shape to the max of
        # target shape so we are sure there is always a crop to be made
        self.scale = A.SmallestMaxSize(max_size=max(out_size), interpolation=0)
        self.out_size = out_size

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Takes a random crop of the image and mask of size `out_size`.
        If image is too small for the crop, this function will scale it up.

        Args:
            image (np.ndarray): color image, (H, W, 3)
            mask (np.ndarray): mask, (H, W, 1)

        Returns:
            image (np.ndarray): color image, (H, W, 3)
            mask (np.ndarray): mask, (H, W, 1)
        """
        if (
            mask.shape[0] < self.out_size[0]
            or mask.shape[1] < self.out_size[1]
        ):  # scale up image if too small
            tsample = self.scale(image=image, mask=mask)
            image, mask = tsample["image"], tsample["mask"]
        tsample = self.aug_fn(
            image=image, mask=mask
        )  # albumentations returns a transformed object
        return tsample["image"], tsample["mask"]


class UniformRandomResize(DualTransform):
    def __init__(
        self,
        scale_range=(0.9, 1.1),
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1,
    ):
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.interpolation = interpolation

    def get_params_dependent_on_targets(self, params):
        scale = (
            torch.rand((1,)).item()
            * (self.scale_range[1] - self.scale_range[0])
            + self.scale_range[0]
        )
        height = int(round(params["image"].shape[0] * scale))
        width = int(round(params["image"].shape[1] * scale))
        return {"new_height": height, "new_width": width}

    def apply(
        self,
        img,
        new_height=0,
        new_width=0,
        interpolation=cv2.INTER_LINEAR,
        **params,
    ):
        # return F.resize(img, height=new_height, width=new_width, interpolation=interpolation)
        return resize.Resize(
            height=new_height, width=new_width, interpolation=interpolation
        )(image=img)["image"]

    def apply_to_keypoint(self, keypoint, new_height=0, new_width=0, **params):
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return "scale_range", "interpolation"

    @property
    def targets_as_params(self):
        return ["image"]


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


def norm_fn(
    x: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    if (xmax := x.max()) != (xmin := x.min()):
        return (x - xmin) / (xmax - xmin)
    if isinstance(x, np.ndarray):
        return np.ones_like(x) * xmax
    if isinstance(x, torch.Tensor):
        return torch.ones_like(x) * xmax
    raise ValueError(f"`x` type is {type(x)} instead of Tensor or ndarray")
