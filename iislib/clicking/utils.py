from typing import Any
from typing import Sequence
from typing import Tuple
from typing import Union

import cv2
import numpy as np
import skimage
import torch
from torch import Tensor
from torch.testing._internal.common_dtype import integral_types


def sample_points_torch(prob_map: torch.Tensor, n_points=1) -> torch.Tensor:
    """Sample point from probability map (using torch)"""
    click_indx = torch.multinomial(prob_map.flatten(), n_points)
    click_coords = torch_unravel_index(
        click_indx, prob_map.squeeze().shape
    )  # squeeze to get (H, W)
    return torch.stack(click_coords, axis=1)  # (n_click, i/j)


def get_largest_incorrect_region(alpha, gt):

    largest_incorrect_BF = []
    for val in [0, 1]:

        incorrect = (gt == val) * (alpha != val)
        ret, labels_con = cv2.connectedComponents(
            incorrect.astype(np.uint8) * 255
        )
        label_unique, counts = np.unique(
            labels_con[labels_con != 0], return_counts=True
        )
        if len(counts) > 0:
            largest_incorrect = labels_con == label_unique[np.argmax(counts)]
            largest_incorrect_BF.append(largest_incorrect)
        else:
            largest_incorrect_BF.append(np.zeros_like(incorrect))

    largest_incorrect_cat = np.argmax(
        [np.count_nonzero(x) for x in largest_incorrect_BF]
    )
    largest_incorrect = largest_incorrect_BF[largest_incorrect_cat]
    return largest_incorrect, largest_incorrect_cat


def get_largest_region(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.ones_like(mask)  # largest false region is anywhere
    labels = skimage.morphology.label(mask, connectivity=1)
    label_unique, counts = np.unique(labels[labels != 0], return_counts=True)
    return 1 * (labels == label_unique[np.argmax(counts)])


def output_target_are_B1HW_in_01(
    outputs: Any = None, targets: Any = None
) -> bool:
    """Return True if inputs are torch Tensors of (B, C, H, W) shape and C=1,
    and if target is binary and output is in [0,1]"""
    targets_ok = targets is None or (
        (len(targets.shape) == 4)
        and (targets.shape[1] == 1)
        and (isinstance(targets, torch.Tensor))
        and ({int(e) for e in targets.unique()}.issubset({0, 1}))
    )
    outputs_ok = outputs is None or (
        (len(outputs.shape) == 4)
        and (outputs.shape[1] == 1)
        and (isinstance(outputs, torch.Tensor))
        and (0 <= outputs.min() and outputs.max() <= 1)
    )
    return targets_ok and outputs_ok


def get_d_prob_map(mask: np.ndarray, hard_thresh: float = 1e-6) -> np.ndarray:
    """
    Get probability map depending on l2 distance to background.

    `hard_thresh` is the excluded external radius in hard thresholding mode
    living in [0,1].
    If `hard_thresh=0` then return distance map (normalized)
    If `hard_thresh=1e-6` then return uniform map
    """
    padded_mask = np.pad(mask, ((1, 1), (1, 1)), "constant")
    dt_map = cv2.distanceTransform(
        padded_mask.astype(np.uint8), cv2.DIST_L2, 0
    )[1:-1, 1:-1]
    inner_mask = (
        dt_map / dt_map.max()
    )  # map distance to [0, 1], 1 is in the center
    inner_mask = (
        hard_thresh < inner_mask if (0 < hard_thresh) else inner_mask
    )  # all ones at hard_thresh from the border
    return inner_mask / max(
        inner_mask.sum(), 1e-6
    )  # return normalized probability


def sample_point(prob_map: np.ndarray) -> np.ndarray:
    """Sample point from probability map"""
    click_indx = int(torch.multinomial(torch.tensor(prob_map.flatten()), 1))
    click_coords = np.unravel_index(click_indx, prob_map.shape)
    return np.array(click_coords)


def get_point_from_mask(
    mask: np.ndarray, hard_thresh: float = 1e-6
) -> np.ndarray:
    """Sample point from inside mask"""
    prob_map = get_d_prob_map(mask, hard_thresh=hard_thresh)
    return sample_point(prob_map)


def positive_erode(mask: np.ndarray, erode_iters: int = 15) -> np.ndarray:
    """Get smaller mask"""
    # mask = np.array(mask.cpu())  # just in case we enter with tensors
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_mask = cv2.erode(
        mask.astype(np.uint8), kernel, iterations=erode_iters
    ).astype(bool)
    return 1 * eroded_mask  # this can be too small, be careful


def positive_dilate(mask: np.ndarray, dilate_iters: int = 15) -> np.ndarray:
    """Get larger mask"""
    # mask = np.array(mask)  # mask is a tensor?
    # expand_r = int(np.ceil(expand_ratio * np.sqrt(mask.sum())))  # old line
    kernel = np.ones((3, 3), np.uint8)
    expanded_mask = cv2.dilate(
        mask.astype(np.uint8), kernel, iterations=dilate_iters
    )
    return 1 * expanded_mask


def safe_erode(
    mask: np.ndarray, erode_iters: int, thresh: float = 0.7
) -> np.ndarray:
    """Erode but maintaining an area at least equal to thresh times the
    original area"""
    masked_area, eroded = mask.sum(), 0
    while (
        np.sum(eroded) <= thresh * masked_area
    ):  # always enters once, reduce erosion iters until eroded mask is big
        # enough
        eroded = positive_erode(mask, erode_iters=erode_iters)
        erode_iters = erode_iters // 2
    assert thresh * masked_area < np.sum(eroded), "Too much erosion!"
    return eroded


def safe_dilate(
    mask: np.ndarray, dilate_iters: int, thresh: float = 1.3
) -> np.ndarray:
    """Dilate but maintaining an area inferior to thresh times the original
    area"""
    masked_area = mask.sum()
    dilated = thresh * masked_area + 1  # init as number
    while thresh * masked_area < np.sum(dilated):  # while too much dilation
        dilated = positive_dilate(mask, dilate_iters=dilate_iters)
        dilate_iters = dilate_iters // 2
    assert dilated.sum() < thresh * masked_area, "Too much dilation!"
    return dilated


def get_outside_border_mask(
    mask: np.ndarray,
    dilate_iters: int = 15,
    safe_thresh: Union[None, float] = 1.3,
) -> Tuple[np.ndarray, bool]:
    """Get border part lying outside mask"""
    if safe_thresh:
        expanded_mask = safe_dilate(
            mask, dilate_iters=dilate_iters, thresh=safe_thresh
        )
    else:
        expanded_mask = positive_dilate(mask, dilate_iters=dilate_iters)
    no_border = np.all(expanded_mask == mask)  # true if there is no border
    expanded_mask[mask.astype(bool)] = 0  # delet contents inside mask
    return 1 * expanded_mask, no_border


# def get_negative_click(mask, near_border=False, uniform_probs=False,
# dilate_iters=15):
#     """Sample a negative click according to the following:
#     `near_border and uniform_probs and (1 < dilate_iters)` means sampling
# uniformly at random from the border
#     `near_border and not uniform_probs` is not allowed
#     `near_border and (dilate_iters==0)` is not allowed
#     `not near_border and uniform_probs and (1 < dilate_iters)` means
# sampling uniformly at random from the background but away of the border
#     `not near_border and not uniform_probs and (1 < dilate_iters)` means
# sampling closer to the center of the background with more probability
#     `not near_border and uniform_probs and (dilate==0)` means sampling
# uniformly from everywhere
#     Note that border is external border of mask.
#     """
#     if mask.size == mask.sum():
#         return None  # mask is everywhere
#     outside_border, no_border = get_outside_border_mask(mask,
# dilate_iters, safe=True)
#     assert (not near_border) or (
#         near_border and uniform_probs
#     ), "`uniform_probs` should be True when sampling `near_border`"
#     if near_border and not no_border:
#         return get_point_from_mask(outside_border)  # get uniformly
# from border
#     elif uniform_probs or no_border:
#         return get_point_from_mask(
#             np.logical_not(mask) - outside_border
#         )  # get uniformly from background
#     else:
#         return get_point_from_mask(
#             np.logical_not(mask) - outside_border, hard_thresh=0
#         )  # get from center of background


# def get_positive_click(mask, near_border=False, uniform_probs=False,
# erode_iters=15):
#     """Sample a positive click according to the following:
#     `near_border and uniform_probs and (1 < erode_iters)` means sampling
# uniformly at random from the border
#     `near_border and not uniform_probs` is not allowed
#     `near_border and (erode_iters==0)` is not allowed
#     `not near_border and uniform_probs and (1 < erode_iters)` means
# sampling uniformly at random from away of the border
#     `not near_border and not uniform_probs and (1 < erode_iters)` means
# sampling closer to the center with more probability
#     `not near_border and uniform_probs and (erode_iters==0)` means
# sampling uniformly from everywhere
#     Note that border is internal border of mask.
#     """
#     if mask.sum() == 0:
#         return None  # no mask from where to sample click
#     eroded, is_same = safe_erode(mask, erode_iters)
#     assert (not near_border) or (
#         near_border and uniform_probs
#     ), "`uniform_probs` should be True when sampling `near_border`"
#     if near_border and not is_same:
#         inside_border = mask - eroded
#         return get_point_from_mask(inside_border)  # get uniformly from
# border
#     elif uniform_probs or is_same:  # near center
#         return get_point_from_mask(eroded)  # get uniformly from center
# region
#     else:
#         return get_point_from_mask(
#             eroded, hard_thresh=0
#         )  # get weighted from center region


# # get_positive_clicks can be made faster by eroding just one time
# def get_positive_clicks_batch(
#     n, masks, near_border=False, uniform_probs=False, erode_iters=15
# ):
#     if n == 0:
#         return [[] for _ in range(masks.shape[0])]
#     elif n < 0:
#         raise ValueError(f"`n` should be positive but is {n}")
#     else:
#         return [
#             [
#                 get_positive_click(
#                     mask[0],
#                     near_border=near_border,
#                     uniform_probs=uniform_probs,
#                     erode_iters=erode_iters,
#                 )
#                 for _ in range(n)
#             ]
#             for mask in masks
#         ]


# def get_negative_clicks(n, mask, near_border, uniform_probs, dilate_iters):
#     raise NotImplementedError(
#         "We will use `get_positive_clicks` from false positive region."
#     )
#     ncs = [
#         get_negative_click(
#             mask,
#             near_border=near_border,
#             uniform_probs=uniform_probs,
#             dilate_iters=dilate_iters,
#         )
#         for _ in range(n)
#     ]
#     return list(filter(lambda x: x is not None, ncs))  # remove None
# # elements


def torch_unravel_index(  # from https://github.com/pytorch/pytorch/pull/66687
    indices: Tensor,
    shape: Union[int, Sequence, Tensor],
    *,
    as_tuple: bool = True,
) -> Union[Tuple[Tensor, ...], Tensor]:
    r"""Converts a `Tensor` of flat indices into a `Tensor` of coordinates
    for the given target shape.
    Args:
        indices: An integral `Tensor` containing flattened indices of a
        `Tensor` of dimension `shape`.
        shape: The shape (can be an `int`, a `Sequence` or a `Tensor`) of
        the `Tensor` for which the flattened `indices` are unraveled.
    Keyword Args:
        as_tuple: A boolean value, which if `True` will return the result
        as tuple of Tensors, else a `Tensor` will be returned. Default: `False`
    Returns:
        unraveled coordinates from the given `indices` and `shape`.
        See description of `as_tuple` for returning a `tuple`.
    .. note:: The default behaviour of this function is analogous to
              `numpy.unravel_index <https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html>`_.
    Example::
        >>> indices = torch.tensor([22, 41, 37])
        >>> shape = (7, 6)
        >>> torch.unravel_index(indices, shape)
        tensor([[3, 4],
                [6, 5],
                [6, 1]])
        >>> torch.unravel_index(indices, shape, as_tuple=True)
        (tensor([3, 6, 6]), tensor([4, 5, 1]))
        >>> indices = torch.tensor([3, 10, 12])
        >>> shape_ = (4, 2, 3)
        >>> torch.unravel_index(indices, shape_, as_tuple=False)
        tensor([[0, 1, 0],
                [1, 1, 1],
                [2, 0, 0]])
    """

    def _helper_type_check(inp: Union[int, Sequence, Tensor], name: str):
        # `indices` is expected to be a tensor, while `shape` can be a sequence/int/tensor
        if name == "shape" and isinstance(inp, Sequence):
            for dim in inp:
                if not isinstance(dim, int):
                    raise TypeError(
                        "Expected shape to have only integral elements."
                    )
                if dim < 0:
                    raise ValueError(
                        "Negative values in shape are not allowed."
                    )
        elif name == "shape" and isinstance(inp, int):
            if inp < 0:
                raise ValueError("Negative values in shape are not allowed.")
        elif isinstance(inp, Tensor):
            if inp.dtype not in integral_types():
                raise TypeError(
                    f"Expected {name} to be an integral tensor, but found dtype: {inp.dtype}"
                )
            if torch.any(inp < 0):
                raise ValueError(f"Negative values in {name} are not allowed.")
        else:
            allowed_types = (
                "Sequence/Scalar (int)/Tensor"
                if name == "shape"
                else "Sequence/Tensor"
            )
            msg = f"{name} should either be a {allowed_types}, but found {type(inp)}"
            raise TypeError(msg)

    _helper_type_check(indices, "indices")
    _helper_type_check(shape, "shape")

    # Convert to a tensor, with the same properties as that of indices
    if isinstance(shape, Sequence):
        shape_tensor: Tensor = indices.new_tensor(shape)
    elif isinstance(shape, int) or (
        isinstance(shape, Tensor) and shape.ndim == 0
    ):
        shape_tensor = indices.new_tensor((shape,))
    else:
        shape_tensor = shape

    # By this time, shape tensor will have dim = 1 if it was passed as scalar (see if-elif above)
    assert (
        shape_tensor.ndim == 1
    ), f"Expected dimension of shape tensor to be <= 1, \
    but got the tensor with dim: {shape_tensor.ndim}."

    # In case no indices passed, return an empty tensor with number of elements = shape.numel()
    if indices.numel() == 0:
        # If both indices.numel() == 0 and shape.numel() == 0, short-circuit to return itself
        shape_numel = shape_tensor.numel()
        if shape_numel == 0:
            raise ValueError(
                "Got indices and shape as empty tensors, expected non-empty tensors."
            )
        output = [indices.new_tensor([]) for _ in range(shape_numel)]
        return tuple(output) if as_tuple else torch.stack(output, dim=1)

    if torch.max(indices) >= torch.prod(shape_tensor):
        raise ValueError("Target shape should cover all source indices.")

    coefs = shape_tensor[1:].flipud().cumprod(dim=0).flipud()
    coefs = torch.cat((coefs, coefs.new_tensor((1,))), dim=0)
    coords = (
        torch.div(indices[..., None], coefs, rounding_mode="trunc")
        % shape_tensor
    )

    if as_tuple:
        return tuple(coords[..., i] for i in range(coords.size(-1)))
    return coords
