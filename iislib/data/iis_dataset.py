from abc import ABC, abstractmethod
import collections
import re
from typing import Any, Tuple
from data.region_selector import dummy
import torch
from torch._six import string_classes
import numpy as np

from data.transforms import norm_fn
from tests.visualization import visualize

class SegDataset(torch.utils.data.Dataset, ABC):
    """Segmentation dataset structure. Load all datasets subclassing the method and
    defining the `get_sample` method. All masks should be given as output in such a
    deterministic function."""

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_sample(self, index:int)-> Tuple(np.ndarray, np.ndarray, Any):
        """returns (image, layers, info)"""

    def at_child_init_end(self):
        '''Call this at child's `__init__` end'''
        assert hasattr(self, 'dataset_samples')
        self.check_sample()

    def check_sample(self):
        sample = self.get_sample(0)
        image, masks, info = sample  # what a sampel should be
        assert (
            image.shape[2] == 3
        ), f"Image should be RGB with channels last but its shape is {image.shape}"
        assert len(masks.shape) == 3, f"Masks should be (H, W, C) but its shape is {masks.shape}"
        assert (
            image.shape[:2] == masks.shape[:2]
        ), "Image and masks should have the same shape but their shapes are {image.shape} and {masks.shape}"

    def __getitem__(self, index:int) -> Tuple(np.ndarray, np.ndarray, Any):
        return self.get_sample(index)

    def __len__(self) -> int:
        return len(self.dataset_samples)



class RegionDatasetWithInfo(torch.utils.data.Dataset):
    """Interactive segmentation dataset, which will yield an image and a target mask when
    queried.
    The image and seed mask come from the `seg_dataset`.
    `region_selector` will create one ground truth mask from the original mask.
    `augmentator` transforms image and ground truth mask simultaneously.
    Output image and mask will be scaled to [0, 1]"""

    def __init__(
        self,
        seg_dataset,
        region_selector,
        augmentator=None,
    ):
        super().__init__()
        self.seg_dataset = seg_dataset
        self.region_selector = region_selector
        self.augmentator = (lambda x: x) if augmentator is None else augmentator

    def __getitem__(self, index: int):
        image, all_masks, info = self.seg_dataset[index]
        target_region = self.region_selector(image, all_masks, info)
        image, target_region = self.augmentator(image, target_region)
        return np.array(norm_fn(image), dtype=float), np.array(norm_fn(target_region), dtype=float), info

    def __len__(self):
        return len(self.seg_dataset)

class RegionDataset(RegionDatasetWithInfo):
    '''
    Exactly the same as `RegionDatasetWithInfo` but returning {'image':image, 'mask':mask} when sampled (C, H, W).
    This is the same output (without 'info' key) that when using `RegionDataloader` over `RegionDatasetWithInfo`
    '''
    def __init__(
        self,
        seg_dataset,
        region_selector,
        augmentator=None,
        debug_visualize=False,
    ):
        super().__init__(seg_dataset, region_selector, augmentator, debug_visualize)
    
    def __getitem__(self, index: int):
        image, mask, info = super().__getitem__(index)
        return {'image':image, 'mask':mask}

class EvaluationDataset(RegionDataset):
    '''Exactly the same as `RegionDataset` but checks if database has binary mask as ground truth.
    Otherwise the evaluation is more complicated and is not implemented.'''
    def __init__(
        self,
        seg_dataset,
        region_selector=dummy,
        augmentator=None,
        debug_visualize=False,
    ):
        assert type(seg_dataset).__name__ in ['BerkeleyDataset', 'GrabCutDataset'], f'{type(seg_dataset).__name__} is not suited for evaluation.'
        assert region_selector == dummy, 'Region selector should be `dummy`'
        super().__init__(seg_dataset, region_selector, augmentator, debug_visualize)
    




######################################################################################################
######################################################################################################
######### all of the things below are to correctly deal with info when dataloading ###################
######################################################################################################
######################################################################################################

### modify DataLoader so that we can get a variable dict in the third output ###
np_str_obj_array_pattern = re.compile(r"[SaUO]")
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)

def my_collate_subfn(batch):
    r"""Copy of `default_collate` treating dicts differently"""
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, collections.abc.Mapping):  # my change
        return [d for d in batch]
    elif isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return my_collate_subfn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(my_collate_subfn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [my_collate_subfn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

def my_collate_fn(batch):
    batch = my_collate_subfn(batch)
    return {'image':batch[0].permute(0, 3, 1, 2), 'mask':batch[1].permute(0, 3, 1, 2), 'info':batch[2]}


class RegionDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=my_collate_fn)

######################################################################################################
######################################################################################################
######### all of the things above are to correctly deal with info when dataloading ###################
######################################################################################################
######################################################################################################



def scin(img):
    """Swap Channels If Needed (scin)"""
    assert img.ndim == 3, "An image has 3 dimensions! (C, H, W) or (H, W, C)"
    if type(img) is torch.Tensor and min(img.shape) == img.shape[0]:
        return img.permute(2, 1, 0)
    elif min(img.shape) == img.shape[-1]:
        return img
    else:
        raise RuntimeError(f"Image has shape {img.shape} which is strange")



