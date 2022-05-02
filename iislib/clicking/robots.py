"""
Robot clicking is defined in this script.
There are robots in increasing complexity level:
implemented:
- robot_01 : randomly samples and sets pos/neg according to target only
- robot_02 : randomly samples from false region only
- robot_03 : randomly samples from largest false region only
to implement:
- robot_04 : distance map sampling
- robot_05 : distance map sampling considering previous clicks
- robot_gto99 : center sampling largest false region considering previous
clicks as in 99% paper
- robot_ritm : sampling as in RITM paper

All the robots 0x do
- Deal with batches
- Sample `n_points`
- A random number of them are negative (if possible)
- Add these points to the previous list
- Encodes the clicks if an encoder is passed needed
"""
from functools import lru_cache
from typing import Callable
from typing import Union

import numpy as np
import torch
from clicking.utils import get_d_prob_map
from clicking.utils import get_largest_region
from clicking.utils import get_outside_border_mask
from clicking.utils import output_target_are_B1HW_in_01
from clicking.utils import safe_erode
from clicking.utils import sample_points_torch
from models.custom.gto99.interaction import click_position
from models.custom.gto99.interaction import get_largest_incorrect_region
from models.custom.ritm.isegm.inference.clicker import Click
from models.custom.ritm.isegm.inference.clicker import Clicker

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
    _pcs: list[list[Point]] = []
    _ncs: list[list[Point]] = []  # (batch element, click)
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
    _pcs: list[list[Point]] = []
    _ncs: list[list[Point]] = []  # (batch element, click)
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
    implementation: detect largest false region and sample randomly from
    inside each one direclty
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
    _pcs: list[list[Point]] = []
    _ncs: list[list[Point]] = []  # (batch element, click)
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


def robot_04(
    outputs: torch.Tensor,  # (B, C, H, W), C=1, contained in [0, 1] (interval)
    targets: torch.Tensor,  # (B, C, H, W), C=1, contained in {0, 1} (set)
    pcs: Clicks,  # indexed by (interaction, batch_element, click)
    ncs: Clicks,
    n_points: int = 1,
    thresh=0.5,
) -> tuple[Clicks, Clicks]:
    """
    Adds n_points into the lists `pcs` and `ncs`
    """
    assert output_target_are_B1HW_in_01(outputs, targets)
    pred_masks = 1 * (thresh < outputs)
    false_masks = torch.logical_xor(pred_masks, targets)
    prob_maps = torch.stack(
        [
            torch.from_numpy(
                get_d_prob_map(false_mask[0].cpu().numpy(), hard_thresh=0)
            )[None, ...]
            for false_mask in false_masks
        ]
    )
    clicks = [
        sample_points_torch(prob_map, n_points) for prob_map in prob_maps
    ]

    is_positive = [
        [targets[b, :, i, j] == 1 for i, j in clicks[b]]
        for b in range(len(clicks))
    ]  # a bool per click
    _pcs: list[list[Point]] = []
    _ncs: list[list[Point]] = []  # (batch element, click)
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


def robot_05(
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
    implementation: detect largest false region and sample randomly from
    inside each one direclty
    """
    assert output_target_are_B1HW_in_01(outputs, targets)
    pred_masks = 1 * (thresh < outputs)
    false_masks = torch.logical_xor(pred_masks, targets)
    clicks = [[]] * len(false_masks)

    for _ in range(n_points):  # create n points
        prob_maps = torch.stack(
            [
                torch.from_numpy(
                    get_d_prob_map(false_mask[0].cpu().numpy(), hard_thresh=0)
                )[None, ...]
                for false_mask in false_masks
            ]  # false_mask and prob_map are both (1, H, W)
        )
        for ind, prob_map in enumerate(prob_maps):
            click = sample_points_torch(prob_map, 1)[0]
            assert len(click) == 2
            false_masks[ind][0, click[0], click[1]] = 0  # background
            clicks[ind].append(click)
    clicks = [torch.stack(click_list) for click_list in clicks]

    is_positive = [
        [targets[b, :, i, j] == 1 for i, j in clicks[b]]
        for b in range(len(clicks))
    ]  # a bool per click
    _pcs: list[list[Point]] = []
    _ncs: list[list[Point]] = []  # (batch element, click)
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


def robot_gto99(
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
    implementation: detect largest false region and sample randomly from
    inside each one direclty
    """
    assert output_target_are_B1HW_in_01(outputs, targets)
    assert n_points == 1, "For getting to 99 paper we use only one click"
    pred_masks = 1 * (thresh < outputs)
    false_masks = torch.logical_xor(pred_masks, targets)
    clicks = [None] * len(false_masks)
    for ind, pred_mask in enumerate(pred_masks):
        target = targets[ind]
        incorrect_region, click_cat = get_largest_incorrect_region(
            pred_mask[0].cpu().numpy(), target[0].cpu().numpy()
        )
        y, x = click_position(incorrect_region, clicks_cat=None)
        clicks[ind] = torch.Tensor(
            [[y, x]]
        ).long()  # we need this because they are indices

    is_positive = [
        [targets[b, :, i, j] == 1 for i, j in clicks[b]]
        for b in range(len(clicks))
    ]  # a bool per click
    _pcs: list[list[Point]] = []
    _ncs: list[list[Point]] = []  # (batch element, click)
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


def robot_ritm(
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
    implementation: detect largest false region and sample randomly from
    inside each one direclty
    """
    assert output_target_are_B1HW_in_01(outputs, targets)
    assert n_points == 1, "For ritm paper clicker we use only one click"
    pred_masks = 1 * (thresh < outputs)
    clicks = [None] * len(targets)  # along batch
    for ind, pred_mask in enumerate(pred_masks):  # along batch
        target = (
            targets[ind][0].cpu().numpy().astype("int32")
        )  # no channel, int32 binary mask
        init_clicks = []
        for interaction_ind, _pcs in enumerate(pcs):  # along interactions
            if 0 < len(_pcs[ind]) and isinstance(_pcs[ind][0], torch.Tensor):
                click = Click(is_positive=True, coords=_pcs[ind][0])
            else:
                click = Click(
                    is_positive=False, coords=ncs[interaction_ind][ind][0]
                )
            init_clicks.append(click)
        clicker = Clicker(gt_mask=target, init_clicks=init_clicks)
        clicker.make_next_click(pred_mask[0].cpu().numpy())
        new_click = clicker.get_clicks()[-1]
        clicks[ind] = torch.Tensor([new_click.coords]).long()

    is_positive = [
        [targets[b, :, i, j] == 1 for i, j in clicks[b]]
        for b in range(len(clicks))
    ]  # a bool per click
    _pcs: list[list[Point]] = []
    _ncs: list[list[Point]] = []  # (batch element, click)
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


###########################################################################
# init robots


def init_robot_no_click(batch_of_targets):
    pcs, ncs = [], []
    return pcs, ncs


def init_robot_random(
    targets: torch.Tensor,  # (B, C, H, W), C=1, contained in {0, 1} (set)
    n_points: int = 1,
) -> tuple[Clicks, Clicks]:
    """
    Creates the lists `pcs` and `ncs` with n_points into them
    Randomly samples and add labels according to target only.
    """
    assert output_target_are_B1HW_in_01(None, targets)
    i_coord = torch.randint(
        targets.shape[-2], (n_points,)
    )  # the same clicks for all elements in batch
    j_coord = torch.randint(targets.shape[-1], (n_points,))
    clicks = list(
        torch.stack((i_coord, j_coord), axis=1)
    )  # (batch_element, coord) or list of points
    is_positive = [
        targets[:, :, i, j] == 1 for i, j in clicks
    ]  # a bool vector per click
    _pcs: list[list[Point]] = []
    _ncs: list[list[Point]] = []  # (batch element, click)
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
    return [_pcs], [
        _ncs
    ]  # (interaction, batch, click) or list of list of list of Points


@lru_cache(maxsize=None)
def generate_probs(max_num_points, gamma):
    probs = []
    last_value = 1
    for _ in range(max_num_points):
        probs.append(last_value)
        last_value *= gamma

    probs = np.array(probs)
    probs /= probs.sum()

    return probs


def init_robot_smartly_random(
    targets: torch.Tensor,  # (B, C, H, W), C=1, contained in {0, 1} (set)
    n_points: int = 1,
) -> tuple[Clicks, Clicks]:
    assert output_target_are_B1HW_in_01(None, targets)
    # sample n_points from an imaginary grid of size n_points x n_points
    # sample n_points noises on i and j
    # sum those
    size_i, size_j = (
        targets.shape[-2] // n_points,
        targets.shape[-1] // n_points,
    )
    flattened_grid_coords = torch.randperm(n_points * n_points)[:n_points]
    noise_i = torch.randint(size_i, (n_points,))
    noise_j = torch.randint(size_j, (n_points,))
    i_coord = flattened_grid_coords // n_points * size_i + noise_i
    j_coord = flattened_grid_coords % n_points * size_j + noise_j

    clicks = list(
        torch.stack((i_coord, j_coord), axis=1)
    )  # (batch_element, coord) or list of points
    is_positive = [
        targets[:, :, i, j] == 1 for i, j in clicks
    ]  # a bool vector per click
    _pcs: list[list[Point]] = []
    _ncs: list[list[Point]] = []  # (batch element, click)
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
    return [_pcs], [
        _ncs
    ]  # (interaction, batch, click) or list of list of list of Points


def init_robot_ritm(
    targets: torch.Tensor,  # (B, C, H, W), C=1, contained in {0, 1} (set)
    n_points: int = 1,
) -> tuple[Clicks, Clicks]:
    """
    Creates the lists `pcs` and `ncs` with n_points into them
    Randomly samples and curates the selected clicks.
    Add labels according to target only.
    """
    # in short, sample some random number of points from the eroded positive,
    # and if first click, it is distance-based.
    # then sample some random number of points from the different negatives
    _pcs, _ncs = [], []

    max_num_points = 12
    prob_gamma = 0.8

    _pos_probs = generate_probs(max_num_points, gamma=prob_gamma)
    for mask in targets:
        # number of positive points
        num_points = 1 + np.random.choice(
            np.arange(max_num_points), p=_pos_probs
        )
        indices = np.argwhere(mask)
        pos_clicks = list(
            indices[np.random.choice(len(indices), num_points, replace=False)]
        )
        # add to list
        _pcs.append(pos_clicks)  # (B, click)

    negative_bg_prob = 0.1
    negative_other_prob = 0.4
    negative_border_prob = 0.5
    neg_strategies = ["bg", "other", "border"]
    neg_strategies_prob = [
        negative_bg_prob,
        negative_other_prob,
        negative_border_prob,
    ]
    _neg_probs = generate_probs(max_num_points + 1, gamma=prob_gamma)

    for mask in targets:
        bg = 1 - mask
        other = bg
        border = get_outside_border_mask(mask)
        required = safe_erode(bg, 4)
        _neg_masks = {
            "bg": bg,
            "other": other,
            "border": border,
            "required": required,
        }

        num_points_required = np.random.choice(
            np.arange(max_num_points + 1), p=_pos_probs
        )
        num_points_strategies = min(
            max_num_points - num_points_required,
            np.random.choice(np.arange(max_num_points + 1), p=_neg_probs),
        )

        indices = np.argwhere(required)
        neg_clicks = list(
            indices[
                np.random.choice(
                    len(indices), num_points_required, replace=False
                )
            ]
        )

        for j in range(num_points_strategies):
            strat = np.random.choice(neg_strategies, p=neg_strategies_prob)
            tmask = _neg_masks[strat]
            indices = np.argwhere(tmask)
            neg_clicks.append(indices[np.random.choice(len(indices), 1)])

        _ncs.append(neg_clicks)

    return [_pcs], [_ncs]


#######################################################################
# explanation of multipointsampler
# sample mask joining masks and eroding them
# get the _neg_probs dict with neg_mask_bg, other, border,
# and neg_masks (eroded)
# sample positive points from selected masks, non negative, with first
# click center
# filter out selected masks if exceeding max_num_points
# sample points for each selected mask
# randomly choose how many points to sample
# if there are probabilities for each mask, put them along the indices
# for j in num_points
# distance-based sampling if first click
# sample a mask according to probs if many masks
# sample from mask if only one mask
# if points sampled from only one mask, take those, fill and return
# else
# if only one first click, take the first points
# sample (unitl max_num_points) from union of masks with given probability
# sample negative points from different masks


######################################################################


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
