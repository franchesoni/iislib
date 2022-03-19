import functools
import skimage

import numpy as np
import torch

## utils ##
def check_input(image, masks, info):
    assert (
        image.ndim == masks.ndim == 3
    ), f"Both image and mask should have three channels instead of {image.ndim} and {masks.ndim}"
    assert (
        min(masks.shape) == masks.shape[2]
    ), f"Mask should be channel last format but its shape is {masks.shape}"  # this works unless image dimensions or number of layers are out of normal range

def get_background_mask(masks):
    return np.max(masks, axis=2) == 0  # region that is 0 in all layers

def get_all_objs(masks):
    # gets layer index and val for each layer and different val and return a list
    return [
        {"layer_ind": layer_ind, "val": val}
        for layer_ind in range(masks.shape[2])
        for val in np.unique(masks[:, :, layer_ind])
    ]

def check_inclusion(mask1, mask2):
    assert set(np.unique(mask1)) == set(np.unique(mask2)) == {0, 1} == {0, 1}, 'Masks are not binary'
    assert not (mask1 == mask2).all(), 'Masks are the same'
    if (np.logical_or(mask1, mask2) == mask1).all():  # contained
        included = True
        parent = mask1
        child = mask2
    elif (np.logical_or(mask1, mask2) == mask2).all():  # contained
        included = True
        parent = mask2
        child = mask1
    else:
        included = False
        parent = None
        child = None
    return included, parent, child


def is_connected(mask):
    return skimage.morphology.label(mask).max() == 1  # only one class

def compute_area(mask):
    assert mask.min() == 0 and mask.max() == 1
    return np.sum(mask == 1)

def get_subregions_as_layers(mask):
    labeled_regions = skimage.morphology.label(mask)
    n_regions = labeled_regions.max()
    layers = [np.zeros_like(mask)] * n_regions
    for l_ind, label in enumerate(range(1, n_regions + 1)):
        layers[l_ind][labeled_regions==label] = 1
    return np.stack(layers)

def get_biggest_region(mask):
    labeled_regions = skimage.morphology.label(mask)
    areas = np.bincount(labeled_regions)
    return np.amax(areas[1:])





## region selectors ##

def dummy(image, masks, info):
    check_input(image, masks, info)
    return 1 * (masks[:, :, 0] == 1)  # only values == 1 from first mask

def random_single(image, masks, info):
    check_input(image, masks, info)
    objs = get_all_objs(masks)
    obj = objs[
        torch.randint(len(objs), (1,))
    ]  # select one object randomly  (use torch because np.random is complicated when parallelizing)
    return 1 * (
        masks[:, :, obj["layer_ind"]] == obj["val"]
    )  # only values from selected layer equal to val

def random_merge(image, masks, info, n_merge=2):
    check_input(image, masks, info)
    objs = get_all_objs(masks)
    sel_objs = np.array(objs)[
        torch.randperm(len(objs))[:n_merge]
    ]  # select objects randomly
    return 1 * np.logical_or([masks[:, :, obj['layer_ind']] == obj['val'] for obj in sel_objs])



