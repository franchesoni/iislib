"""
Robot clicking is defined in this script.
There are robots in increasing complexity level:
implemented:
- robot_01 : randomly samples and sets pos/neg according to target only
- robot_02 : randomly samples from false region only
- robot_03 : randomly samples from largest false region only
to implement:
- robot_04 : randomly samples from largest false region considering
previous clicks
- robot_05 : distance map sampling considering previous clicks
- robot_99 : center sampling largest false region considering previous
clicks as in 99% paper
- robot_ritm : sampling as in RITM paper

All the robots 0x do
- Deal with batches
- Sample `n_points`
- A random number of them are negative (if possible)
- Add these points to the previous list
- Encodes the clicks if an encoder is passed needed
"""
from typing import Callable
from typing import Union

import torch
from clicking.utils import get_largest_region
from clicking.utils import output_target_are_B1HW_in_01
from clicking.utils import sample_points_torch

# import sys
# sys.path.append("/home/franchesoni/iis/iislib/tests/")
# from visualization import visualize

Point = Union[list[int], torch.Tensor]
Clicks = list[
    list[list[Point]]
]  # axes : (interaction, batch_element, click_number)


def build_robot_mix(
    list_of_robots: list[Callable], robots_weights: list[float]
):
    assert len(list_of_robots) == len(robots_weights)
    sw = sum(robots_weights)
    robots_weights = torch.tensor(
        [w / sw for w in robots_weights]
    )  # make probability dist

    def robot_mix(*args, **kwargs):
        robot_ind = torch.multinomial(robots_weights, 1)
        return list_of_robots[robot_ind](*args, **kwargs)

    return robot_mix


def robot_01(
    outputs: torch.Tensor,  # (B, C, H, W), C=1, contained in [0, 1] (interval)
    targets: torch.Tensor,  # (B, C, H, W), C=1, contained in {0, 1} (set)
    pcs: Clicks,  # indexed by (interaction, batch_element, click)
    ncs: Clicks,
    n_points: int = 1,
) -> tuple[Clicks, Clicks]:
    """
    Adds n_points into the lists `pcs` and `ncs`
    Randomly samples according to target only.
    """
    assert output_target_are_B1HW_in_01(outputs, targets)
    i_coord = torch.randint(
        outputs.shape[-2], (n_points,)
    )  # the same clicks for all elements in batch
    j_coord = torch.randint(outputs.shape[-1], (n_points,))
    clicks = list(
        torch.stack((i_coord, j_coord), axis=1)
    )  # (batch_element, coord) or list of points
    is_positive = [
        targets[:, :, i, j] == 1 for i, j in clicks
    ]  # a bool vector per click
    _pcs, _ncs = [], []  # (batch element, click)
    for ind_target in range(
        len(targets)
    ):  # add the clicks for each image in the batch
        _pcs.append([])
        _ncs.append([])
        for ind_click, click in enumerate(clicks):
            if is_positive[ind_click][ind_target]:
                _pcs[-1].append(click)
            else:
                _ncs[-1].append(click)
    pcs.append(_pcs)  # nested list indexed as follows:
    ncs.append(
        _ncs
    )  # (interaction, batch, click) or list of list of list of Points
    return pcs, ncs


def robot_02(
    outputs: torch.Tensor,  # (B, C, H, W), C=1, contained in [0, 1] (interval)
    targets: torch.Tensor,  # (B, C, H, W), C=1, contained in {0, 1} (set)
    pcs: Clicks,  # indexed by (interaction, batch_element, click)
    ncs: Clicks,
    n_points: int = 1,
    thresh=0.5,
) -> tuple[Clicks, Clicks]:
    """
    Adds n_points into the lists `pcs` and `ncs`
    Randomly samples from false region.
    implementation: sample randomly from inside each mask direclty
    """
    assert output_target_are_B1HW_in_01(outputs, targets)
    pred_masks = 1 * (thresh < outputs)
    false_masks = torch.logical_xor(pred_masks, targets)
    prob_maps = (
        false_masks / false_masks.sum((2, 3))[..., None, None]
    )  # (B, 1, H, W) / (B, 1, 1, 1)
    clicks = [
        sample_points_torch(prob_map, n_points) for prob_map in prob_maps
    ]
    is_positive = [
        [targets[b, :, i, j] == 1 for i, j in clicks[b]]
        for b in range(len(clicks))
    ]  # a bool per click
    _pcs, _ncs = [], []  # (batch element, click)
    for b in range(len(targets)):  # add the clicks for each image in the batch
        _pcs.append([])
        _ncs.append([])
        for ind_click, click in enumerate(clicks[b]):
            if is_positive[b][ind_click]:
                _pcs[-1].append(click)
            else:
                _ncs[-1].append(click)
    pcs.append(_pcs)  # nested list indexed as follows:
    ncs.append(
        _ncs
    )  # (interaction, batch, click) or list of list of list of Points
    return pcs, ncs


def robot_03(
    outputs: torch.Tensor,  # (B, C, H, W), C=1, contained in [0, 1] (interval)
    targets: torch.Tensor,  # (B, C, H, W), C=1, contained in {0, 1} (set)
    pcs: Clicks,  # indexed by (interaction, batch_element, click)
    ncs: Clicks,
    n_points: int = 1,
    thresh=0.5,
) -> tuple[Clicks, Clicks]:
    """
    Adds n_points into the lists `pcs` and `ncs`
    Randomly samples from largest false region.
    implementation: detect largest false region and sample randomly from inside each one direclty
    """
    assert output_target_are_B1HW_in_01(outputs, targets)
    pred_masks = 1 * (thresh < outputs)
    false_masks = torch.logical_xor(pred_masks, targets)
    largest_false_masks = torch.stack(
        [
            torch.from_numpy(get_largest_region(false_mask.cpu()))
            for false_mask in false_masks
        ]
    )  # done one cpu :sad:
    prob_maps = (
        largest_false_masks / largest_false_masks.sum((2, 3))[..., None, None]
    )  # (B, 1, H, W) / (B, 1, 1, 1)
    clicks = [
        sample_points_torch(prob_map, n_points) for prob_map in prob_maps
    ]

    is_positive = [
        [targets[b, :, i, j] == 1 for i, j in clicks[b]]
        for b in range(len(clicks))
    ]  # a bool per click
    _pcs, _ncs = [], []  # (batch element, click)
    for b in range(len(targets)):  # add the clicks for each image in the batch
        _pcs.append([])
        _ncs.append([])
        for ind_click, click in enumerate(clicks[b]):
            if is_positive[b][ind_click]:
                _pcs[-1].append(click)
            else:
                _ncs[-1].append(click)
    pcs.append(_pcs)  # nested list indexed as follows:
    ncs.append(
        _ncs
    )  # (interaction, batch, click) or list of list of list of Points
    return pcs, ncs


# def get_next_points_1(
#     prev_output,
#     gt_mask,
#     n_points=None,
#     prev_pc_mask=None,
#     prev_nc_mask=None,
#     is_first_iter=False,
#     n_positive=None,
# ):
#     """Simple point getter.
#     - Adds positive and negative clicks on a random number.
#     - If first iteration, add only positive clicks
#     - Sample points from erroneous regions (false positive or false
# negative regions)
#     - Sample positive points closer to the center
#     - Sample negative points uniformly
#     - Add these points to previous point masks
#     """
#     # intialize prev clicks at zero if needed
#     prev_pc_mask = (
#         torch.zeros_like(gt_mask) if prev_pc_mask is None else prev_pc_mask
#     )
#     prev_nc_mask = (
#         torch.zeros_like(gt_mask) if prev_nc_mask is None else prev_nc_mask
#     )

#     pos_region = 1 * (
#         prev_output < gt_mask
#     )  # false negative region, i.e. when you predicted 0 but mask was 1
#     neg_region = 1 * (
#         gt_mask < prev_output
#     )  # false positive region, i.e. you predicted 1 but mask was 0

#     if is_first_iter:  # only positive points at first iteration
#         n_positive = n_points
#     elif n_positive is None:  # sample how many positive
#         n_positive = torch.randint(
#             n_points + 1, (1,)
#         )  # anything in [0, n_points]  CORRECT  # how many of the clicks
# # are positive vs negative

#     pos_clicks = get_positive_clicks_batch(n_positive, pos_region)
#     neg_clicks = get_positive_clicks_batch(
#         n_points - n_positive,
#         neg_region,
#         near_border=False,
#         uniform_probs=True,
#         erode_iters=0,
#     )
#     pc_mask = disk_mask_from_coords_batch(pos_clicks, prev_pc_mask)
#     nc_mask = (
#         disk_mask_from_coords_batch(neg_clicks, prev_nc_mask)
#         if neg_clicks
#         else torch.zeros_like(pc_mask)
#     )

#     return (
#         pc_mask[:, None, :, :],
#         nc_mask[:, None, :, :],
#         pos_clicks,
#         neg_clicks,
#     )
