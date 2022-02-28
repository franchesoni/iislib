import collections
import re
import torch
from torch._six import string_classes
import matplotlib.pyplot as plt


def scin(img):
    """Swap Channels If Needed (scin)"""
    assert img.ndim == 3, "An image has 3 dimensions! (C, H, W) or (H, W, C)"
    if type(img) is torch.Tensor and min(img.shape) == img.shape[0]:
        return img.permute(2, 1, 0)
    elif min(img.shape) == img.shape[-1]:
        return img
    else:
        raise RuntimeError(f"Image has shape {img.shape} which is strange")


def visualize(img, name):
    """Save image as png"""
    plt.figure()
    plt.imshow(img)
    plt.axis("off")
    plt.grid()
    plt.savefig(name + ".png")
    plt.close()


class SegDataset:
    """Segmentation dataset structure. Load all datasets subclassing the method and
    defining the `get_sample` method. All masks should be given as output in such a
    deterministic function."""

    def __init__(self):
        self.dataset_samples = (
            None  # dataset should be loaded on the init of the child class
        )

    def get_sample(self, index):
        raise NotImplementedError(
            "You should write this for your specific dataset, this is only a placeholder."
        )
        return image, masks, info  # expected output

    def check_sample(self):
        sample = self.get_sample(0)
        image, masks, info = sample
        assert (
            image.shape[2] == 3
        ), f"Image should be RGB with channels last but its shape is {image.shape}"
        assert (
            image.shape[:2] == masks[:2]
        ), "Image and masks should have the same shape but their shapes are {image.shape} and {masks.shape}"

    def __getitem__(self, index):
        return self.get_sample(index)

    def __len__(self):
        return len(self.dataset_samples)


class RegionDataset:
    """Interactive segmentation dataset, which will yield an image and a target mask when
    queried.
    The image and seed mask come from the `seg_dataset`.
    `region_selector` will create one ground truth mask from the original mask.
    `augmentator` transforms image and ground truth mask simultaneously."""

    def __init__(
        self,
        seg_dataset,
        region_selector,
        augmentator=None,
        debug_visualize=False,
    ):
        self.seg_dataset = seg_dataset
        self.region_selector = region_selector
        self.augmentator = (lambda x: x) if augmentator is None else augmentator
        self.debug_visualize = debug_visualize

    def get_sample(self, index: int):
        image, all_masks, info = self.seg_dataset.get_sample(index)
        if self.debug_visualize:
            visualize(image, "orig_image")
            for layer_ind in range(all_masks.shape[-1]):
                visualize(all_masks[:, :, layer_ind], f"layer_{layer_ind}")
        target_region = self.region_selector(image, all_masks, info)
        image, target_region = self.augmentator(image, target_region)
        return image, target_region, info

    def __getitem__(self, index):
        return self.get_sample(index)

    def __len__(self):
        return len(self.seg_dataset)


### modify DataLoader so that we can get a variable dict in the third output ###
np_str_obj_array_pattern = re.compile(r"[SaUO]")
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def my_collate_fn(batch):
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

            return my_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(my_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [my_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


class RegionDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, collate_fn=my_collate_fn)


### test things in here and more ###
def test():
    from data.datasets.coco_lvis import CocoLvisDataset
    from data.transformations import RandomCrop
    from data.region_selector import dummy, random_single

    import pytorch_lightning as pl

    pl.seed_everything(0)

    seg_dataset = CocoLvisDataset("/home/franchesoni/adisk/iis_datasets/datasets/LVIS")
    region_selector = random_single
    augmentator = RandomCrop(out_size=(224, 224))
    iis_dataset = RegionDataset(seg_dataset, region_selector, augmentator)
    iis_dataloader = RegionDataLoader(iis_dataset, batch_size=2, num_workers=0)
    for ind, batch in enumerate(iis_dataloader):
        print(ind)
        images, masks, infos = batch
        visualize(images[0], "image")
        visualize(masks[0], "mask")
        breakpoint()


if __name__ == "__main__":
    test()
