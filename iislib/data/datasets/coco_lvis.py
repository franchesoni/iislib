import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Tuple
from typing import Union

import cv2
import numpy as np
from data.iis_dataset import SegDataset


def drop_from_first_if_not_in_second(
    alist: list,
    blist: list,
    fn_a: Union[Callable, None] = None,
    fn_b: Union[Callable, None] = None,
) -> Tuple[list, list]:
    """this works for sorted lists over which we apply fn_a and fn_b \
       (and remain sorted)"""
    assert isinstance(alist, list) and isinstance(blist, list)
    assert alist == sorted(alist) and blist == sorted(
        blist
    ), "input lists should be sorted"
    if fn_a is None:

        def fn_a(x):
            return x  # identity

    if fn_b is None:

        def fn_b(x):
            return x  # identity

    aa = [fn_a(e) for e in alist]
    bb = [fn_b(e) for e in blist]

    assert aa == sorted(set(aa)) and bb == sorted(
        set(bb)
    ), "resulting lists after fn should be of unique elements and be sorted"

    indices_to_remove = []
    ia, ib = 0, 0
    while ia < len(aa) and ib < len(bb):
        if bb[ib] < aa[ia]:  # b is smaller, advance ib until it's not anymore
            ib += 1
        elif aa[ia] == bb[ib]:  # they are equal, it's ok, advance a
            ia += 1
        elif (
            aa[ia] < bb[ib]
        ):  # b became greater because a is not in blist, remove a
            indices_to_remove.append(ia)
            ia += 1

    if ib == len(bb):
        # the blist ended: at the previous ib, b was smaller than a, which
        # means that a and all the following should be removed
        while ia < len(aa):
            indices_to_remove.append(ia)
            ia += 1

    return [
        e for ind, e in enumerate(alist) if ind not in indices_to_remove
    ], indices_to_remove


class CocoLvisDataset(SegDataset):
    """Tricky dataset.
    It seems to have images and masks which are, for some reason,
    divided into layers (I would say that this is to allow superpositions
    but who knows).
    Each mask layer can have many integer values or classes.
    There is also extra info: `num_instance_masks`, `hierarchy`, and
    `objs_mapping`. Presumably,
    the `num_instance_masks` does not include stuff segmentations.
    `objs_mapping` seems to
    include all possible masks as tuples `(layer_number, class_number)`,
    and hierarchy assigns
    parents and childs.
    Stuff is defined as those objects in `objs_mapping` that are not in
    between the first
    `num_instance_masks`."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        split: str = "train",
        anno_file: str = "hannotation.pickle",
    ):
        super().__init__()
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self.split = split
        self._images_path = self._split_path / "images"
        self._masks_path = self._split_path / "masks"
        with open(self._split_path / anno_file, "rb") as f:
            self.dataset_samples = sorted(pickle.load(f).items())
        # filter out those names that are not present in the actual data
        self.available_images = sorted(os.listdir(self._images_path))
        self.dataset_samples, _ = drop_from_first_if_not_in_second(
            self.dataset_samples,
            self.available_images,
            fn_a=lambda dsample: dsample[0] + ".jpg",
        )
        self.at_child_init_end()

        # # to inspect what we are doing, uncomment this part below
        # ds, removed_indices = drop_from_first_if_not_in_second(
        #     self.dataset_samples,
        #     self.available_images,
        #     fn_a=lambda dsample: dsample[0] + ".jpg",
        # )
        # removed_entries = [
        #     e
        #     for i, e in enumerate(self.dataset_samples)
        #     if i in removed_indices
        # ]

    def get_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray, Any]:
        image_id, sample = self.dataset_samples[index]
        image_path = self._images_path / f"{image_id}.jpg"

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        packed_masks_path = self._masks_path / f"{image_id}.pickle"
        with open(packed_masks_path, "rb") as f:
            encoded_layers, objs_mapping = pickle.load(f)
        layers = [
            cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers
        ]
        layers = np.stack(layers, axis=2)

        instances_info = deepcopy(sample["hierarchy"])
        for inst_id, inst_info in list(instances_info.items()):
            if inst_info is None:
                inst_info = {"children": [], "parent": None, "node_level": 0}
                instances_info[inst_id] = inst_info
            inst_info["mapping"] = objs_mapping[inst_id]
        info = {
            "hierachy": instances_info,
            "num_instance_masks": sample["num_instance_masks"],
        }

        # return `image`, masks in `layers`, and `info`. `info` has
        # as keys `['hierarchy', 'num_instance_masks']`.
        return image, layers, info
