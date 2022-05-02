import torch


def interact(
    model,
    init_z,
    init_y,
    robot_clicker,
    init_robot_clicker,
    batch,
    interaction_steps=None,
    max_interactions=None,
    clicks_per_step=1,
    max_init_clicks=5,
    batch_idx=None,
):
    assert bool(interaction_steps) != bool(
        max_interactions
    ), "deterministic or max, can't be both"
    if interaction_steps is None:
        assert (
            max_interactions
        ), "This should be an int if `interaction_steps` is None"
        interaction_steps_m1 = torch.randint(
            max_interactions, (1,)
        )  # we don't add 1 because that 1 is the step in which gradients
        # are computed (outside the for loop)
    else:
        interaction_steps_m1 = interaction_steps - 1
    init_clicks = int(
        torch.randint(max_init_clicks, (1,))
    )  # cast to int because this is used by torch.randperm

    image, target = (
        batch["image"],
        batch["mask"],
    )

    assert isinstance(image, torch.Tensor) and isinstance(
        target, torch.Tensor
    ), "`image` and `target` should be torch tensors"
    assert (
        image.shape[1] == 3
    ), f"Image should be (B, 3, H, W) but is {image.shape}"
    assert (
        target.shape[1] == 1
    ), f"Target should be (B, 1, H, W) but is {target.shape}"
    z = init_z(image, target)
    y = init_y(image, target)
    if init_robot_clicker is not None and 0 < init_clicks:
        pcs, ncs = init_robot_clicker(target, init_clicks)
    else:
        pcs, ncs = [], []

    with torch.no_grad():
        for iter_indx in range(interaction_steps_m1):  # isteps-1 times
            pcs, ncs = robot_clicker(
                y, target, n_points=clicks_per_step, pcs=pcs, ncs=ncs
            )
            y, z = model(image, z, pcs, ncs)
    # last interaction with gradient computation
    pcs, ncs = robot_clicker(
        y, target, n_points=clicks_per_step, pcs=pcs, ncs=ncs
    )
    y, z = model(image, z, pcs, ncs)
    return y, z, pcs, ncs


# def interact_single_step(
#     is_first_iter,
#     model,
#     image,
#     gt_mask,
#     prev_output=None,
#     n_clicks=1,
#     pc_mask=None,
#     nc_mask=None,
#     pcs=[],
#     ncs=[],
# ):
#     image = (
#         image[None].permute(0, 3, 1, 2)
#         if image.shape[-1] == 3
#         else image[None]
#     )
#     gt_mask = (
#         gt_mask[None].permute(0, 3, 1, 2)
#         if gt_mask.shape[-1] == 1
#         else gt_mask[None]
#     )
#     if prev_output is None:
#         prev_output = torch.zeros_like(image[:, :1])

#     with torch.no_grad():
#         pc_mask, nc_mask, _pcs, _ncs = get_next_points_1(
#             n_points=1,
#             prev_output=1 * (0.5 < prev_output),
#             gt_mask=gt_mask,
#             prev_pc_mask=pc_mask,
#             prev_nc_mask=nc_mask,
#             is_first_iter=is_first_iter,
#         )
#         pcs.append(_pcs)
#         ncs.append(_ncs)
#         x, aux = image, torch.cat(
#             (pc_mask, nc_mask, prev_output), dim=1
#         )  # image and aux input, passed separately because preprocessing
#         output = torch.sigmoid(
#             model(x, aux)
#         )  # actual computation of gradients
#     return output, pcs, ncs, pc_mask, nc_mask
