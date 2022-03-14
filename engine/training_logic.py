from code import interact
import torch
import segmentation_models_pytorch as smp

from clicking.robots import get_next_points_1  # deprecate

from data.iis_dataset import visualize
from models.wrappers.iis_smp_wrapper import EarlySMP



def interact_single_step(is_first_iter, model, image, gt_mask, prev_output=None, n_clicks=1, pc_mask=None, nc_mask=None, pcs=[], ncs=[]):
    image = image[None].permute(0, 3, 1, 2) if image.shape[-1] == 3 else image[None]
    gt_mask = gt_mask[None].permute(0, 3, 1, 2) if gt_mask.shape[-1] == 1 else gt_mask[None]
    if prev_output is None:
        prev_output = torch.zeros_like(image[:, :1])

    with torch.no_grad():
        pc_mask, nc_mask, _pcs, _ncs = get_next_points_1(
            n_points=1,
            prev_output=1 * (0.5 < prev_output),
            gt_mask=gt_mask,
            prev_pc_mask=pc_mask,
            prev_nc_mask=nc_mask,
            is_first_iter=is_first_iter,
        )
        pcs.append(_pcs)
        ncs.append(_ncs)
        x, aux = image, torch.cat(
            (pc_mask, nc_mask, prev_output), dim=1
        )  # image and aux input, passed separately because preprocessing
        output = torch.sigmoid(model(x, aux))  # actual computation of gradients
    return output, pcs, ncs, pc_mask, nc_mask
 


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
