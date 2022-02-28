import torch
import numpy as np
import cv2


def is_connected(mask):
    raise NotImplementedError


def compute_area(mask):
    raise NotImplementedError


def get_subregions_as_layers(mask):
    raise NotImplementedError


def get_biggest_region(mask):
    raise NotImplementedError


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
    mask = np.array(mask)
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
    while np.sum(eroded) < thresh * masked_area:
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


def test():
    from data.datasets.coco_lvis import CocoLvisDataset
    from data.transformations import RandomCrop
    from data.region_selector import dummy, random_single
    from data.iis_dataset import RegionDataLoader, RegionDataset, visualize, scin

    import pytorch_lightning as pl

    pl.seed_everything(0)

    iis_dataloader = RegionDataLoader(
        RegionDataset(
            seg_dataset=CocoLvisDataset(
                "/home/franchesoni/adisk/iis_datasets/datasets/LVIS"
            ),
            region_selector=random_single,
            augmentator=RandomCrop(out_size=(224, 224)),
        ),
        batch_size=2,
        num_workers=0,
    )

    def get_positive_clicks(n, mask, near_border, uniform_probs, erode_iters):
        return [
            get_positive_click(
                masks[0], near_border=True, uniform_probs=True, erode_iters=5
            )
            for _ in range(10)
        ]

    def get_negative_clicks(n, mask, near_border, uniform_probs, dilate_iters):
        ncs = [
            get_negative_click(
                masks[0], near_border=True, uniform_probs=True, dilate_iters=15
            )
            for _ in range(10)
        ]
        return list(filter(lambda x: x is not None, ncs))  # remove None elements

    for ind, batch in enumerate(iis_dataloader):
        print(f"index {ind}")
        # if ind == 29:
        #     breakpoint()
        images, masks, infos = batch
        masks = np.array(masks.squeeze())
        pcs = get_positive_clicks(
            10, masks[0], near_border=True, uniform_probs=True, erode_iters=5
        )
        ncs = get_negative_clicks(
            10, masks[0], near_border=True, uniform_probs=True, dilate_iters=15
        )

        visualize_clicks(images[0], masks[0], 0.3, pcs, ncs, "vis1")
        pcs = get_positive_clicks(
            100, masks[0], near_border=False, uniform_probs=True, erode_iters=15
        )
        ncs = get_negative_clicks(
            100, masks[0], near_border=False, uniform_probs=True, dilate_iters=15
        )
        visualize_clicks(images[0], masks[0], 0.3, pcs, ncs, "vis2")
        pcs = get_positive_clicks(
            100, masks[0], near_border=False, uniform_probs=False, erode_iters=15
        )
        ncs = get_negative_clicks(
            100, masks[0], near_border=False, uniform_probs=False, dilate_iters=15
        )
        visualize_clicks(images[0], masks[0], 0.3, pcs, ncs, "vis3")
        pcs = get_positive_clicks(
            100, masks[0], near_border=False, uniform_probs=True, erode_iters=0
        )
        ncs = get_negative_clicks(
            100, masks[0], near_border=False, uniform_probs=True, dilate_iters=0
        )
        visualize_clicks(images[0], masks[0], 0.3, pcs, ncs, "vis4")
        breakpoint()


if __name__ == "__main__":
    test()
