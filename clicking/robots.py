"""
Robot clicking is defined in this script.
There are robots in increasing complexity level:
- robot_01 : randomly samples according to target only
- robot_02 : randomly samples from false region only
- robot_03 : randomly samples from largest false region only
- robot_04 : randomly samples from largest false region considering previous clicks
- robot_05 : distance map sampling considering previous clicks
- robot_99 : center sampling largest false region considering previous clicks as in 99% paper
- robot_ritm : sampling as in RITM paper 

All the robots 0x do
- Deal with batches
- Sample `n_points`
- A random number of them are negative (if possible)
- Add these points to the previous list
- Encodes the clicks if an encoder is passed needed
"""

import torch

from clicking.utils import check_masks


def robot_01(
    outputs,  # (B, C, H, W)
    targets,
    n_points=1,
    pcs=[],  # indexed by (interaction, batch_element, click) 
    ncs=[],

):
    """Randomly samples according to target only"""
    check_masks(outputs, targets)
    i_coord = torch.randint(outputs.shape[-2], (n_points,))
    j_coord = torch.randint(outputs.shape[-1], (n_points,))
    clicks = zip(i_coord, j_coord)  # same for all elements in batch
    is_positive = [targets[:, :, i, j] == 1 for i, j in clicks]  # a bool vector per click

    _pcs, _ncs = [], []  # (batch element, click)
    for ind_target in range(len(targets)):  # add for each image the positive clicks
        _pcs.append([])
        _ncs.append([])
        for ind_click, click in enumerate(clicks):
            if is_positive[ind_click][ind_target]:
                _pcs[-1].append(click)
            else:
                _ncs[-1].append(click)

    pcs.append(_pcs)  # nested list indexed as follows:
    ncs.append(_ncs)  # (interaction, batch, click)

    return pcs, ncs





def get_next_points_1(
    prev_output,
    gt_mask,
    n_points=None,
    prev_pc_mask=None,
    prev_nc_mask=None,
    is_first_iter=False,
    n_positive=None,
):
    """Simple point getter.
    - Adds positive and negative clicks on a random number.
    - If first iteration, add only positive clicks
    - Sample points from erroneous regions (false positive or false negative regions)
    - Sample positive points closer to the center
    - Sample negative points uniformly
    - Add these points to previous point masks
    """
    # intialize prev clicks at zero if needed
    prev_pc_mask = torch.zeros_like(gt_mask) if prev_pc_mask is None else prev_pc_mask
    prev_nc_mask = torch.zeros_like(gt_mask) if prev_nc_mask is None else prev_nc_mask

    pos_region = 1 * (
        prev_output < gt_mask
    )  # false negative region, i.e. when you predicted 0 but mask was 1
    neg_region = 1 * (
        gt_mask < prev_output
    )  # false positive region, i.e. you predicted 1 but mask was 0

    if is_first_iter:  # only positive points at first iteration
        n_positive = n_points
    elif n_positive is None:  # sample how many positive
        n_positive = torch.randint(
            n_points + 1, (1,)
        )  # anything in [0, n_points]  CORRECT  # how many of the clicks are positive vs negative

    pos_clicks = get_positive_clicks_batch(n_positive, pos_region)
    neg_clicks = get_positive_clicks_batch(
        n_points - n_positive,
        neg_region,
        near_border=False,
        uniform_probs=True,
        erode_iters=0,
    )
    pc_mask = disk_mask_from_coords_batch(pos_clicks, prev_pc_mask)
    nc_mask = (
        disk_mask_from_coords_batch(neg_clicks, prev_nc_mask)
        if neg_clicks
        else torch.zeros_like(pc_mask)
    )

    return (
        pc_mask[:, None, :, :],
        nc_mask[:, None, :, :],
        pos_clicks,
        neg_clicks,
    )
