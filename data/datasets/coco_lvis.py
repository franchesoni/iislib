from pathlib import Path
import pickle
import random
import numpy as np
import cv2
from copy import deepcopy

from data.iis_dataset import SegDataset


class CocoLvisDataset(SegDataset):
    """Tricky dataset.
    It seems to have images and masks which are, for some reason,
    divided into layers (I would say that this is to allow superpositions but who knows).
    Each mask layer can have many integer values or classes.
    There is also extra info: `num_instance_masks`, `hierarchy`, and `objs_mapping`. Presumably,
    the `num_instance_masks` does not include stuff segmentations. `objs_mapping` seems to
    include all possible masks as tuples `(layer_number, class_number)`, and hierarchy assigns
    parents and childs.
    Stuff is defined as those objects in `objs_mapping` that are not in between the first
    `num_instance_masks`."""

    def __init__(
        self,
        dataset_path,
        split="train",
        stuff_prob=0.0,
        anno_file="hannotation.pickle",
    ):
        dataset_path = Path(dataset_path)
        self._split_path = dataset_path / split
        self.split = split
        self._images_path = self._split_path / "images"
        self._masks_path = self._split_path / "masks"
        with open(self._split_path / anno_file, "rb") as f:
            self.dataset_samples = sorted(pickle.load(f).items())

    def get_sample(self, index):
        image_id, sample = self.dataset_samples[index]
        image_path = self._images_path / f"{image_id}.jpg"

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        packed_masks_path = self._masks_path / f"{image_id}.pickle"
        with open(packed_masks_path, "rb") as f:
            encoded_layers, objs_mapping = pickle.load(f)
        layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
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

        """return `image`, masks in `layers`, and `info`. `info` has as keys `['hierarchy', 'num_instance_masks']`."""
        return image, layers, info


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = CocoLvisDataset(
        "/home/franchesoni/adisk/iis_datasets/datasets/LVIS"
    )
    for ind in range(100):
        sample = dataset.get_sample(ind)
        assert (
            dataset.get_sample(ind)[0] == sample[0]
        ).all()  # check that it works the same always
        image, layers, info = sample
        plt.imshow(image)
        plt.show()
        plt.savefig("image.png")
        for idx, layer in enumerate(np.moveaxis(layers, 2, 0)):
            plt.imshow(layer)
            plt.show()
            plt.savefig(f"layer_{idx}.png")
        print(info)
        breakpoint()


# removed

# # probability of not erasing the last masks which are not counted as instances (background stuff)
# self.stuff_prob = stuff_prob
# # if not erasing, these things are set to not have parents


# if self.stuff_prob > 0 and random.random() < self.stuff_prob:
#     for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
#         instances_info[inst_id] = {
#             'mapping': objs_mapping[inst_id],
#             'parent': None,
#             'children': []
#         }
# else:
#     for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
#         layer_indx, mask_id = objs_mapping[inst_id]
#         layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0
