from code import interact
import torch
import segmentation_models_pytorch as smp

from data.clicking import get_positive_clicks_batch, disk_mask_from_coords_batch

from data.iis_dataset import visualize
from models.iis_smp_wrapper import EarlySMP


def get_next_points_1(
    n_points,
    prev_output,
    gt_mask,
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
        n_positive = (
            torch.randint(n_points+1, (1,))  # anything in [0, n_points]  CORRECT
        )  # how many of the clicks are positive vs negative

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


def interact(
    model, batch, interaction_steps=None, max_interactions=None, clicks_per_step=1, batch_idx=None
):
    if interaction_steps is None:
        assert max_interactions, "This should be an int if `interaction_steps` is None"
        interaction_steps = torch.randint(
            max_interactions, (1,)
        )  # we don't substract 1 because that 1 is the step in which gradients are computed (outside the for loop)

    image, gt_mask = (
        batch["image"],
        batch["mask"],
    )
    image = image.permute(0, 3, 1, 2) if image.shape[-1] == 3 else image
    gt_mask = gt_mask.permute(0, 3, 1, 2) if gt_mask.shape[-1] == 1 else gt_mask 

    orig_image, orig_gt_mask = (
        image.clone(),
        gt_mask.clone(),
    )

    prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
    pc_mask, nc_mask, pcs, ncs = None, None, [], []

    with torch.no_grad():
        for iter_indx in range(interaction_steps):
            pc_mask, nc_mask, _pcs, _ncs = get_next_points_1(
                n_points=clicks_per_step,
                prev_output=1 * (0.5 < prev_output),
                gt_mask=orig_gt_mask,
                prev_pc_mask=pc_mask,
                prev_nc_mask=nc_mask,
                is_first_iter=iter_indx == 0,
            )
            pcs.append(_pcs)
            ncs.append(_ncs)
            x, aux = image, torch.cat(
                (pc_mask, nc_mask, prev_output), dim=1
            )  # image and aux input, passed separately because preprocessing
            prev_output = torch.sigmoid(model(x, aux))

        pc_mask, nc_mask, _pcs, _ncs = get_next_points_1(
            clicks_per_step,
            1 * (0.5 < prev_output),
            orig_gt_mask,
            pc_mask,
            nc_mask,
            is_first_iter=iter_indx == 0,
        )
        pcs.append(_pcs)
        ncs.append(_ncs)
        x, aux = image, torch.cat(
            (pc_mask, nc_mask, prev_output), dim=1
        )  # image and aux input, passed separately because preprocessing
    output = torch.sigmoid(model(x, aux))  # actual computation of gradients
    return output, pc_mask, nc_mask, pcs, ncs
