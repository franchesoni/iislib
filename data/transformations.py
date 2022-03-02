from copy import deepcopy 
from torchvision.transforms import CenterCrop
import albumentations as A
# (iis) franchesoni@weird-power:~/iis/iis_framework$ numpy cv2 matplotlib torch albumentations torchvision

class RandomCrop:
    def __init__(self, out_size=None):
        self.aug_fn = A.CropNonEmptyMaskIfExists(*out_size)
        self.scale = A.SmallestMaxSize(max_size=max(out_size), interpolation=0)  # if image is small we take the min of current shape to the max of target shape so we are sure there is always a crop to be made
        self.out_size = out_size

    def __call__(self, image, mask):
        if mask.shape[0] < self.out_size[0] or mask.shape[1] < self.out_size[1]:  # scale up image if too small
            tsample = self.scale(image=image, mask=mask)
            image, mask = tsample['image'], tsample['mask']
        tsample = self.aug_fn(image=image, mask=mask)  # albumentations returns a transformed object
        return tsample['image'], tsample['mask'][:, :, None]  # add one dim


class Dummy:
    def __init__(self, out_size=None):
        assert out_size is None

    def __call__(self, image, mask):
        return image, mask
        

# ########### ALBUMENTATIONS ############
# from albumentations.core.serialization import SERIALIZABLE_REGISTRY
# from albumentations import ImageOnlyTransform, ReplayCompose

# def remove_image_only_transforms(sdict):
#     if not 'transforms' in sdict:
#         return sdict

#     keep_transforms = []
#     for tdict in sdict['transforms']:
#         cls = SERIALIZABLE_REGISTRY[tdict['__class_fullname__']]
#         if 'transforms' in tdict:
#             keep_transforms.append(remove_image_only_transforms(tdict))
#         elif not issubclass(cls, ImageOnlyTransform):
#             keep_transforms.append(tdict)
#     sdict['transforms'] = keep_transforms

#     return sdict


# def augment(image, mask, augmentator):
#     aug_output = augmentator(image=image, mask=mask)
#     image = aug_output['image']
#     mask = aug_output['mask']
#     remove_small_objects(image, mask, min_area=1)
#     return image, mask


# ########## GENERAL ###########


# def remove_small_objects(image, mask, min_area=1):

#     if self._objects and not 'area' in list(self._objects.values())[0]:
#         self._compute_objects_areas()

#     for obj_id, obj_info in list(self._objects.items()):
#         if obj_info['area'] < min_area:
#             self._remove_object(obj_id)


#     def _compute_objects_areas(self):
#         inverse_index = {node['mapping']: node_id for node_id, node in self._objects.items()}
#         ignored_regions_keys = set(self._ignored_regions)

#         for layer_indx in range(self._encoded_masks.shape[2]):
#             objects_ids, objects_areas = get_labels_with_sizes(self._encoded_masks[:, :, layer_indx])
#             for obj_id, obj_area in zip(objects_ids, objects_areas):
#                 inv_key = (layer_indx, obj_id)
#                 if inv_key in ignored_regions_keys:
#                     continue
#                 try:
#                     self._objects[inverse_index[inv_key]]['area'] = obj_area
#                     del inverse_index[inv_key]
#                 except KeyError:
#                     layer = self._encoded_masks[:, :, layer_indx]
#                     layer[layer == obj_id] = 0
#                     self._encoded_masks[:, :, layer_indx] = layer

#         for obj_id in inverse_index.values():
#             self._objects[obj_id]['area'] = 0

# def _remove_object(self, obj_id):
#     obj_info = self._objects[obj_id]
#     obj_parent = obj_info['parent']
#     for child_id in obj_info['children']:
#         self._objects[child_id]['parent'] = obj_parent

#     if obj_parent is not None:
#         parent_children = self._objects[obj_parent]['children']
#         parent_children = [x for x in parent_children if x != obj_id]
#         self._objects[obj_parent]['children'] = parent_children + obj_info['children']

#     del self._objects[obj_id]

# def get_bbox_from_mask(mask):
#     rows = np.any(mask, axis=1)
#     cols = np.any(mask, axis=0)
#     rmin, rmax = np.where(rows)[0][[0, -1]]
#     cmin, cmax = np.where(cols)[0][[0, -1]]

#     return rmin, rmax, cmin, cmax


# def expand_bbox(bbox, expand_ratio, min_crop_size=None):
#     rmin, rmax, cmin, cmax = bbox
#     rcenter = 0.5 * (rmin + rmax)
#     ccenter = 0.5 * (cmin + cmax)
#     height = expand_ratio * (rmax - rmin + 1)
#     width = expand_ratio * (cmax - cmin + 1)
#     if min_crop_size is not None:
#         height = max(height, min_crop_size)
#         width = max(width, min_crop_size)

#     rmin = int(round(rcenter - 0.5 * height))
#     rmax = int(round(rcenter + 0.5 * height))
#     cmin = int(round(ccenter - 0.5 * width))
#     cmax = int(round(ccenter + 0.5 * width))

#     return rmin, rmax, cmin, cmax


# def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
#     return (max(rmin, bbox[0]), min(rmax, bbox[1]),
#             max(cmin, bbox[2]), min(cmax, bbox[3]))


# def get_bbox_iou(b1, b2):
#     h_iou = get_segments_iou(b1[:2], b2[:2])
#     w_iou = get_segments_iou(b1[2:4], b2[2:4])
#     return h_iou * w_iou


# def get_segments_iou(s1, s2):
#     a, b = s1
#     c, d = s2
#     intersection = max(0, min(b, d) - max(a, c) + 1)
#     union = max(1e-6, max(b, d) - min(a, c) + 1)
#     return intersection / union

# def get_labels_with_sizes(x):  # assign a label to each value in x and count how many times the value is repeated. Return labels with positive count, except label 0.
#     obj_sizes = np.bincount(x.flatten())  # count number of repetitions for each value
#     labels = np.nonzero(obj_sizes)[0].tolist()  # get indices of nonzero values
#     labels = [x for x in labels if x != 0]
#     return labels, obj_sizes[labels].tolist()

# def compute_objects_areas(objs_mapping):
#     inverse_index = {node['mapping']: node_id for node_id, node in self._objects.items()}

#     for layer_indx in range(self._encoded_masks.shape[2]):
#         objects_ids, objects_areas = get_labels_with_sizes(self._encoded_masks[:, :, layer_indx])
#         for obj_id, obj_area in zip(objects_ids, objects_areas):
#             inv_key = (layer_indx, obj_id)
#             try:
#                 self._objects[inverse_index[inv_key]]['area'] = obj_area
#                 del inverse_index[inv_key]
#             except KeyError:
#                 layer = self._encoded_masks[:, :, layer_indx]
#                 layer[layer == obj_id] = 0
#                 self._encoded_masks[:, :, layer_indx] = layer

#     for obj_id in inverse_index.values():
#         self._objects[obj_id]['area'] = 0