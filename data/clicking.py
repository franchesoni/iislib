import torch
import numpy as np
import copy
import cv2


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


def draw_disk_from_coords_single(point, initial_mask, radius=5):
    """points is a `torch.Tensor` of size (2,)"""
    assert {int(e) for e in set(torch.unique(initial_mask))}.issubset({0, 1})
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
    out_mask = copy.copy(prev_mask)
    for point in points:
        out_mask = draw_disk_from_coords_single(point, out_mask, radius=radius)
    return out_mask


def disk_mask_from_coords_batch(pointss, prev_masks, radius=5):
    out_masks = []
    for ind in range(len(pointss)):
        points, prev_mask = pointss[ind], prev_masks[ind][0]  # from one batch
        out_mask = prev_mask.clone()
        for point in points:
            out_mask = draw_disk_from_coords_single(point, out_mask, radius=radius)
        out_masks.append(out_mask)
    return torch.stack(out_masks)


def test_disk_mask_from_coords():
    import matplotlib.pyplot as plt

    points = [(130, 120), (200, 20)]
    out_shape = (256, 256)
    mask = disk_mask_from_coords(points, out_shape)
    plt.imshow(mask)
    plt.savefig("temp.png")


def get_d_prob_map(mask, hard_thresh=1e-6):
    """Get probability map depending on l2 distance to background.
    `hard_thresh` is the excluded radius in hard mode.
    If `hard_thresh=0` then return normalized distance map
    If `hard_thresh=1e-6` then return uniform map
    """
    padded_mask = np.pad(mask, ((1, 1), (1, 1)), "constant")
    dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
    inner_mask = dt / dt.max()
    inner_mask = hard_thresh < inner_mask if (0 < hard_thresh) else inner_mask
    return inner_mask / max(inner_mask.sum(), 1e-6)


def sample_point(prob_map):
    """Sample point from probability map"""
    click_indx = int(torch.multinomial(torch.tensor(prob_map.flatten()), 1))
    click_coords = np.unravel_index(click_indx, prob_map.shape)
    return np.array(click_coords)


def get_point_from_mask(mask, hard_thresh=1e-6):
    """Sample point from inside mask"""
    prob_map = get_d_prob_map(mask, hard_thresh=hard_thresh)
    return sample_point(prob_map)


def positive_erode(mask, erode_iters=15):
    """Get smaller mask"""
    mask = np.array(mask.cpu())
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_mask = cv2.erode(
        mask.astype(np.uint8), kernel, iterations=erode_iters
    ).astype(bool)
    return 1 * eroded_mask  # this can be too small, be careful


def positive_dilate(mask, dilate_iters=15):
    """Get larger mask"""
    mask = np.array(mask)
    # expand_r = int(np.ceil(expand_ratio * np.sqrt(mask.sum())))  # old line
    kernel = np.ones((3, 3), np.uint8)
    expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=dilate_iters)
    return 1 * expanded_mask


def get_outside_border_mask(mask, dilate_iters=15, safe=True):
    """Get border from outside mask"""
    mask = np.array(mask)
    if not safe:
        expanded_mask = positive_dilate(mask, dilate_iters=dilate_iters)
    else:
        expanded_mask, no_border = safe_dilate(mask, dilate_iters=dilate_iters)
    # expand_r = int(np.ceil(expand_ratio * np.sqrt(mask.sum())))  # old line
    expanded_mask[mask.astype(bool)] = 0
    return 1 * expanded_mask, no_border


def safe_erode(mask, erode_iters, thresh=0.7):
    """Erode but maintaining an area at least equal to thresh times the original area"""
    masked_area, eroded = mask.sum(), 0
    while np.sum(eroded) <= thresh * masked_area:
        eroded = positive_erode(mask, erode_iters=erode_iters)
        erode_iters = erode_iters // 2
    assert thresh * masked_area < eroded.sum(), "Too much erosion!"
    return eroded, np.all(eroded == mask)


def safe_dilate(mask, dilate_iters, thresh=1.3):
    """Dilate but maintaining an area inferior to thresh times the original area"""
    masked_area = mask.sum()
    dilated = masked_area * thresh + 1  # init as number
    while thresh * masked_area < np.sum(dilated):  # too much dilation
        dilated = positive_dilate(mask, dilate_iters=dilate_iters)
        dilate_iters = dilate_iters // 2
    assert dilated.sum() < thresh * masked_area, "Too much dilation!"
    return dilated, np.all(dilated == mask)


def get_negative_click(mask, near_border=False, uniform_probs=False, dilate_iters=15):
    """Sample a negative click according to the following:
    `near_border and uniform_probs and (1 < dilate_iters)` means sampling uniformly at random from the border
    `near_border and not uniform_probs` is not allowed
    `near_border and (dilate_iters==0)` is not allowed
    `not near_border and uniform_probs and (1 < dilate_iters)` means sampling uniformly at random from the background but away of the border
    `not near_border and not uniform_probs and (1 < dilate_iters)` means sampling closer to the center of the background with more probability
    `not near_border and uniform_probs and (dilate==0)` means sampling uniformly from everywhere
    Note that border is external border of mask.
    """
    if mask.size == mask.sum():
        return None  # mask is everywhere
    outside_border, no_border = get_outside_border_mask(mask, dilate_iters, safe=True)
    assert (not near_border) or (
        near_border and uniform_probs
    ), "`uniform_probs` should be True when sampling `near_border`"
    if near_border and not no_border:
        return get_point_from_mask(outside_border)  # get uniformly from border
    elif uniform_probs or no_border:
        return get_point_from_mask(
            np.logical_not(mask) - outside_border
        )  # get uniformly from background
    else:
        return get_point_from_mask(
            np.logical_not(mask) - outside_border, hard_thresh=0
        )  # get from center of background


def get_positive_click(mask, near_border=False, uniform_probs=False, erode_iters=15):
    """Sample a positive click according to the following:
    `near_border and uniform_probs and (1 < erode_iters)` means sampling uniformly at random from the border
    `near_border and not uniform_probs` is not allowed
    `near_border and (erode_iters==0)` is not allowed
    `not near_border and uniform_probs and (1 < erode_iters)` means sampling uniformly at random from away of the border
    `not near_border and not uniform_probs and (1 < erode_iters)` means sampling closer to the center with more probability
    `not near_border and uniform_probs and (erode_iters==0)` means sampling uniformly from everywhere
    Note that border is internal border of mask.
    """
    if mask.sum() == 0:
        return None  # no mask from where to sample click
    eroded, is_same = safe_erode(mask, erode_iters)
    assert (not near_border) or (
        near_border and uniform_probs
    ), "`uniform_probs` should be True when sampling `near_border`"
    if near_border and not is_same:
        inside_border = mask - eroded
        return get_point_from_mask(inside_border)  # get uniformly from border
    elif uniform_probs or is_same:  # near center
        return get_point_from_mask(eroded)  # get uniformly from center region
    else:
        return get_point_from_mask(
            eroded, hard_thresh=0
        )  # get weighted from center region


norm_fn = lambda x: (x - x.min()) / (x.max() - x.min())


def visualize_clicks(image, mask, alpha, pc_list, nc_list, name):
    import matplotlib.pyplot as plt

    image = norm_fn(np.array(image))
    mean_color = image.sum((0, 1)) / image.size
    min_rgb = np.argmin(mean_color)
    mask_color = np.ones((1, 1, 3)) * np.eye(3)[min_rgb][None, None, :]
    out = (
        image / image.max() * (1 - alpha) + mask[:, :, None] * mask_color * alpha
    )  # add inverted mean color mask
    plt.figure()
    plt.imshow(out)
    plt.axis("off")
    plt.grid()
    for pc in pc_list:
        plt.scatter(pc[1], pc[0], s=10, color="g")
    for nc in nc_list:
        plt.scatter(nc[1], nc[0], s=10, color="r")
    plt.savefig(name + ".png")
    plt.close()


# get_positive_clicks can be made faster by eroding just one time
def get_positive_clicks_batch(
    n, masks, near_border=False, uniform_probs=False, erode_iters=15
):
    if n == 0:
        return [[] for _ in range(masks.shape[0])]
    elif n < 0:
        raise ValueError(f'`n` should be positive but is {n}')
    else:
        return [
            [
                get_positive_click(
                    mask[0],
                    near_border=near_border,
                    uniform_probs=uniform_probs,
                    erode_iters=erode_iters,
                )
                for _ in range(n)
            ]
            for mask in masks  
        ]



def get_negative_clicks(n, mask, near_border, uniform_probs, dilate_iters):
    raise NotImplementedError(
        "We will use `get_positive_clicks` from false positive region."
    )
    ncs = [
        get_negative_click(
            mask,
            near_border=near_border,
            uniform_probs=uniform_probs,
            dilate_iters=dilate_iters,
        )
        for _ in range(n)
    ]
    return list(filter(lambda x: x is not None, ncs))  # remove None elements

