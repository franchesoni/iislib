import functools
import skimage

import numpy as np
import torch

## utils ##
@functools.cache
def generate_probs(max_num_points, gamma):  # keep one more point with P = gamma
    p = gamma ** np.arange(max_num_points)
    return p / sum(p)

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





# remove_small_objects
# get_object_mask
# compute_objects_area


# class MultiPointSampler(BasePointSampler):
#     def __init__(self, max_num_points, prob_gamma=0.7, expand_ratio=0.1,
#                  positive_erode_prob=0.9, positive_erode_iters=3,
#                  negative_bg_prob=0.1, negative_other_prob=0.4, negative_border_prob=0.5,
#                  merge_objects_prob=0.0, max_num_merged_objects=2,
#                  use_hierarchy=False, soft_targets=False,
#                  first_click_center=False, only_one_first_click=False,
#                  sfc_inner_k=1.7, sfc_full_inner_prob=0.0):
#         super().__init__()
#         self.max_num_points = max_num_points
#         self.expand_ratio = expand_ratio
#         self.positive_erode_prob = positive_erode_prob
#         self.positive_erode_iters = positive_erode_iters
#         self.merge_objects_prob = merge_objects_prob
#         self.use_hierarchy = use_hierarchy
#         self.soft_targets = soft_targets
#         self.first_click_center = first_click_center
#         self.only_one_first_click = only_one_first_click
#         self.sfc_inner_k = sfc_inner_k
#         self.sfc_full_inner_prob = sfc_full_inner_prob

#         if max_num_merged_objects == -1:  # if negative set to maxnumpoints
#             max_num_merged_objects = max_num_points
#         self.max_num_merged_objects = max_num_merged_objects

#         self.neg_strategies = ['bg', 'other', 'border']
#         self.neg_strategies_prob = [negative_bg_prob, negative_other_prob, negative_border_prob]
#         assert math.isclose(sum(self.neg_strategies_prob), 1.0)

#         self._pos_probs = generate_probs(max_num_points, gamma=prob_gamma)
#         self._neg_probs = generate_probs(max_num_points + 1, gamma=prob_gamma)
#         self._neg_masks = None


#     # merge
#     # sample from inside children <- to do this we need the hierarchy
#     def sample_object(self, sample):  # :Dsample
#         gt_mask, pos_masks, neg_masks = self._sample_mask(sample)
#         binary_gt_mask = gt_mask > 0.5 if self.soft_targets else gt_mask > 0

#         self.selected_mask = gt_mask
#         self._selected_masks = pos_masks

#         neg_mask_bg = np.logical_not(binary_gt_mask)
#         neg_mask_border = self._get_border_mask(binary_gt_mask)
#         if len(sample) <= len(self._selected_masks):
#             neg_mask_other = neg_mask_bg
#         else:
#             neg_mask_other = np.logical_and(np.logical_not(sample.get_background_mask()),
#                                             np.logical_not(binary_gt_mask))

#         self._neg_masks = {
#             'bg': neg_mask_bg,
#             'other': neg_mask_other,
#             'border': neg_mask_border,
#             'required': neg_masks
#         }

#     def _sample_mask(self, sample):  # (take it)
#         root_obj_ids = sample.root_objects

#         if len(root_obj_ids) > 1 and random.random() < self.merge_objects_prob:
#             max_selected_objects = min(len(root_obj_ids), self.max_num_merged_objects)
#             num_selected_objects = np.random.randint(2, max_selected_objects + 1)
#             random_ids = random.sample(root_obj_ids, num_selected_objects)
#         else:
#             random_ids = [random.choice(root_obj_ids)]

#         gt_mask = None
#         pos_segments = []
#         neg_segments = []
#         for obj_id in random_ids:
#             obj_gt_mask, obj_pos_segments, obj_neg_segments = self._sample_from_masks_layer(obj_id, sample)
#             if gt_mask is None:
#                 gt_mask = obj_gt_mask
#             else:
#                 gt_mask = np.maximum(gt_mask, obj_gt_mask)

#             pos_segments.extend(obj_pos_segments)
#             neg_segments.extend(obj_neg_segments)

#         pos_masks = [self._positive_erode(x) for x in pos_segments]
#         neg_masks = [self._positive_erode(x) for x in neg_segments]

#         return gt_mask, pos_masks, neg_masks


#     def _sample_from_masks_layer(self, obj_id, sample):
#         objs_tree = sample._objects

#         if not self.use_hierarchy:
#             node_mask = sample.get_object_mask(obj_id)
#             gt_mask = sample.get_soft_object_mask(obj_id) if self.soft_targets else node_mask
#             return gt_mask, [node_mask], []

#         def _select_node(node_id):
#             node_info = objs_tree[node_id]
#             if not node_info['children'] or random.random() < 0.5:
#                 return node_id
#             return _select_node(random.choice(node_info['children']))

#         selected_node = _select_node(obj_id)
#         node_info = objs_tree[selected_node]
#         node_mask = sample.get_object_mask(selected_node)
#         gt_mask = sample.get_soft_object_mask(selected_node) if self.soft_targets else node_mask
#         pos_mask = node_mask.copy()

#         return gt_mask, [pos_mask], negative_segments

#     def _positive_erode(self, mask):  # take it
#         if random.random() > self.positive_erode_prob:
#             return mask

#         kernel = np.ones((3, 3), np.uint8)
#         eroded_mask = cv2.erode(mask.astype(np.uint8),
#                                 kernel, iterations=self.positive_erode_iters).astype(np.bool)

#         assert eroded_mask.max() == 1, 'check the number below, mask is not binary'
#         if eroded_mask.sum() > 10:  # if eroded mask is large enough
#             return eroded_mask
#         else:
#             return mask

#     def _get_border_mask(self, mask):
#         expand_r = int(np.ceil(self.expand_ratio * np.sqrt(mask.sum())))
#         kernel = np.ones((3, 3), np.uint8)
#         expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=expand_r)
#         expanded_mask[mask.astype(np.bool)] = 0
#         return expanded_mask



# def get_point_candidates(obj_mask, k=1.7, full_prob=0.0):
#     if full_prob > 0 and random.random() < full_prob:
#         return obj_mask

#     padded_mask = np.pad(obj_mask, ((1, 1), (1, 1)), 'constant')

#     dt = cv2.distanceTransform(padded_mask.astype(np.uint8), cv2.DIST_L2, 0)[1:-1, 1:-1]
#     if k > 0:
#         inner_mask = dt > dt.max() / k
#         return np.argwhere(inner_mask)
#     else:
#         prob_map = dt.flatten()
#         prob_map /= max(prob_map.sum(), 1e-6)
#         click_indx = np.random.choice(len(prob_map), p=prob_map)
#         click_coords = np.unravel_index(click_indx, dt.shape)
#         return np.array([click_coords])

# this may be useful as an heuristic
        # # just sample negative from parent
        # negative_segments = []
        # if node_info['parent'] is not None and node_info['parent'] in objs_tree:
        #     parent_mask = sample.get_object_mask(node_info['parent'])
        #     negative_segments.append(np.logical_and(parent_mask, np.logical_not(node_mask)))  

        # # all of this is to not sample from children
        # for child_id in node_info['children']:  # if child is small don't sample from inside it
        #     if objs_tree[child_id]['area'] / node_info['area'] < 0.10:
        #         child_mask = sample.get_object_mask(child_id)
        #         pos_mask = np.logical_and(pos_mask, np.logical_not(child_mask))

        # if node_info['children']:
        #     max_disabled_children = min(len(node_info['children']), 3)
        #     num_disabled_children = np.random.randint(0, max_disabled_children + 1)
        #     disabled_children = random.sample(node_info['children'], num_disabled_children)

        #     for child_id in disabled_children:
        #         child_mask = sample.get_object_mask(child_id)
        #         pos_mask = np.logical_and(pos_mask, np.logical_not(child_mask))
        #         if self.soft_targets:
        #             soft_child_mask = sample.get_soft_object_mask(child_id)
        #             gt_mask = np.minimum(gt_mask, 1.0 - soft_child_mask)
        #         else:
        #             gt_mask = np.logical_and(gt_mask, np.logical_not(child_mask))
        #         negative_segments.append(child_mask)

