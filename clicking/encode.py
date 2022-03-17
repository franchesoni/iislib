import copy

import numpy as np
import torch

def encode_clicks(pcs, ncs,
    encoding_fn=None,
    pos_encoding=None,
    neg_encoding=None):
    '''
    `pcs` is a nested list indexed by (interaction, batch_element, click)
    `ncs` is a nested list indexed by (interaction, batch_element, click)
    the last list inside `pcs` or `ncs` corresponds to the new click(s) to encode.
    `pos_encoding`, `neg_encoding` are the previous encodings to be overwritten.
    '''
    if encoding_fn:
        pos_encoding, neg_encoding = encoding_fn(pcs, ncs, pos_encoding, neg_encoding)
    return pos_encoding, neg_encoding

def encode_disks(pcs, ncs, pos_encoding, neg_encoding, radius=5):
    """see documentation of `encode_clicks`"""
    _pcs, _ncs = pcs[-1], ncs[-1]  # (batch_element, click)
    pos_encoding = disk_mask_from_coords_batch(_pcs, pos_encoding, radius=radius)
    neg_encoding = disk_mask_from_coords_batch(_ncs, neg_encoding, radius=radius)
    return pos_encoding, neg_encoding

def np_simplest_disk(radius):
    out_side = radius * 2 + 1
    out = np.zeros((out_side, out_side))
    rows, cols = np.mgrid[:out_side, :out_side]
    dist_to_center = np.sqrt((rows - radius) ** 2 + (cols - radius) ** 2)
    return 1 * (dist_to_center <= radius)

def simplest_disk(radius):
    out_side = radius * 2 + 1
    out = torch.zeros((out_side, out_side))
    rows, cols = torch.meshgrid([torch.arange(out_side)]*2, indexing='ij')
    dist_to_center = torch.sqrt((rows - radius) ** 2 + (cols - radius) ** 2)
    return 1 * (dist_to_center <= radius)


def draw_disk_from_single_coords(point, initial_mask, radius=5):
    """points is a `torch.Tensor` of size (2,)"""
    assert len(initial_mask.shape) == 2, '`initial_mask` should be (H, W)'
    assert {int(e) for e in set(torch.unique(initial_mask))}.issubset({0, 1}), '`initial_mask` should be binary'
    if point is None:  # do nothing
        return initial_mask
    row_ind, col_ind = point[0], point[1]
    out_mask = initial_mask.clone()
    top_offset = max(radius - row_ind, 0)  # put disk considering this left offset
    left_offset = max(radius - col_ind, 0)  # put disk considering this top offset
    bottom_offset = max(
        radius - (out_mask.shape[0] - row_ind - 1), 0
    )  # put disk considering this right offset
    right_offset = max(
        radius - (out_mask.shape[1] - col_ind - 1), 0
    )  # put disk considering this bottom offset
    out_mask[
        row_ind - radius + top_offset : row_ind + radius + 1 - bottom_offset,
        col_ind - radius + left_offset : col_ind + radius + 1 - right_offset,
    ] = torch.logical_or(
        out_mask[
            row_ind - radius + top_offset : row_ind + radius + 1 - bottom_offset,
            col_ind - radius + left_offset : col_ind + radius + 1 - right_offset,
        ],
        simplest_disk(radius)[
            top_offset : 2 * radius + 1 - bottom_offset,
            left_offset : 2 * radius + 1 - right_offset,
        ].to(out_mask.device),  # added this to put this new tensor on the same device 
    )
    return out_mask


def disk_mask_from_coords(points, prev_mask, radius=5):
    out_mask = prev_mask.clone() if type(prev_mask) == torch.Tensor else copy.copy(prev_mask)
    for point in points:
        out_mask = draw_disk_from_single_coords(point, out_mask, radius=radius)
    return out_mask


def disk_mask_from_coords_batch(pointss, prev_masks, radius=5):
    out_masks = []
    for ind in range(len(pointss)):
        points, prev_mask = pointss[ind], prev_masks[ind][0]  # from one batch
        out_mask = disk_mask_from_coords(points, prev_mask, radius)[None, ...]
        out_masks.append(out_mask)
    return torch.stack(out_masks)


