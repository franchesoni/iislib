from typing import Union

import numpy as np
import torch

Point = Union[list[int], torch.Tensor]


# Disks
def np_simplest_disk(radius: int) -> np.ndarray:
    """Creates a binary matrix containing a disk of the given radius using
    numpy"""
    out_side = radius * 2 + 1
    rows, cols = np.mgrid[:out_side, :out_side]
    dist_to_center = np.sqrt((rows - radius) ** 2 + (cols - radius) ** 2)
    return 1 * (dist_to_center <= radius)


def torch_simplest_disk(radius: int) -> torch.Tensor:
    """Creates a binary matrix containing a disk of the given radius
    using torch"""
    out_side = radius * 2 + 1
    rows, cols = torch.meshgrid([torch.arange(out_side)] * 2, indexing="ij")
    dist_to_center = torch.sqrt((rows - radius) ** 2 + (cols - radius) ** 2)
    return 1 * (dist_to_center <= radius)


def draw_disk_from_single_coords(
    point: Point, initial_mask: torch.Tensor, radius: int = 5
) -> torch.Tensor:
    """Adds disk of given `radius` centered in `point` to `initial_mask`
    `initial_mask` should be a binary ({0, 1}) matrix
    `points` should be a `torch.Tensor` of size (2,) with
    (row_ind, col_ind)"""
    assert len(initial_mask.shape) == 2, "`initial_mask` should be (H, W)"
    assert {int(e) for e in set(torch.unique(initial_mask))}.issubset(
        {0, 1}
    ), "`initial_mask` should be binary"
    if point is None:  # do nothing
        return initial_mask
    row_ind, col_ind = point[0], point[1]
    out_mask = initial_mask.clone()
    top_offset = max(radius - row_ind, 0)  # to consider borders
    left_offset = max(radius - col_ind, 0)
    bottom_offset = max(radius - (out_mask.shape[0] - row_ind - 1), 0)
    right_offset = max(radius - (out_mask.shape[1] - col_ind - 1), 0)
    out_mask[
        row_ind - radius + top_offset : row_ind + radius + 1 - bottom_offset,
        col_ind - radius + left_offset : col_ind + radius + 1 - right_offset,
    ] = torch.logical_or(  # add disk in position considering border clipping
        out_mask[
            row_ind
            - radius
            + top_offset : row_ind
            + radius
            + 1
            - bottom_offset,
            col_ind
            - radius
            + left_offset : col_ind
            + radius
            + 1
            - right_offset,
        ],
        torch_simplest_disk(radius)[
            top_offset : 2 * radius + 1 - bottom_offset,
            left_offset : 2 * radius + 1 - right_offset,
        ].to(
            out_mask.device
        ),  # added this to move the disk tensor on the same device
    )
    return out_mask


def disk_mask_from_coords(
    points: list[Point], prev_mask: torch.Tensor, radius: int = 5
) -> torch.Tensor:
    """Adds disks of given `radius` centered in `points` list to
    `prev_mask`"""
    out_mask = prev_mask.clone()
    for point in points:
        out_mask = draw_disk_from_single_coords(point, out_mask, radius=radius)
    return out_mask


def disk_mask_from_coords_batch(
    pointss: list[list[Point]], prev_masks: torch.Tensor, radius: int = 5
) -> torch.Tensor:
    """Calls `disk_mask_from_coords` for each pair of `points` list and
    `prev_mask` in `zip(pointss, prev_masks)`"""
    assert (
        len(pmshape := prev_masks.shape) == 4 and pmshape[1] == 1
    ), f"`prev_masks` should be (B, 1, H, W) but is {pmshape}"
    assert len(pointss) == len(
        prev_masks
    ), "Batch size should be consistent across inputs"
    return torch.stack(
        [
            disk_mask_from_coords(pointss[ind], prev_masks[ind][0], radius)[
                None, ...
            ]
            for ind in range(len(pointss))
        ]
    )


def encode_disks_last_clicks(
    pcs: list[list[list[Point]]],
    ncs: list[list[list[Point]]],
    pos_encoding: list[torch.Tensor],
    neg_encoding: list[torch.Tensor],
    radius: int = 5,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    `pcs` is a nested list indexed by (interaction, batch_element, click)
    `ncs` is a nested list indexed by (interaction, batch_element, click)
    the last list inside `pcs` or `ncs` refers to the new click(s) to encode.
    `pos_encoding`, `neg_encoding` are previous encodings to be overwritten.
    `last_xcs` will have nested dim : (batch_element, click)
    """
    last_pcs, last_ncs = pcs[-1], ncs[-1]
    pos_encoding = disk_mask_from_coords_batch(
        last_pcs, pos_encoding, radius=radius
    )
    neg_encoding = disk_mask_from_coords_batch(
        last_ncs, neg_encoding, radius=radius
    )
    return pos_encoding, neg_encoding


def encode_disks_from_scratch(
    pcs: list[list[list[Point]]],
    ncs: list[list[list[Point]]],
    pos_encoding: list[torch.Tensor],
    neg_encoding: list[torch.Tensor],
    radius: int = 5,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    `pcs` is a nested list indexed by (interaction, batch_element, click)
    `ncs` is a nested list indexed by (interaction, batch_element, click)
    the last list inside `pcs` or `ncs` refers to the new click(s) to encode.
    `pos_encoding`, `neg_encoding` are previous encodings to be overwritten.
    """
    for _pcs, _ncs in zip(pcs, ncs):  # of dim (batch_element, click)
        pos_encoding, neg_encoding = encode_disks_last_clicks(
            [_pcs], [_ncs], pos_encoding, neg_encoding, radius
        )
    return pos_encoding, neg_encoding


# Disks end
