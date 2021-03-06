from pathlib import Path
from typing import Any
from typing import Tuple
from typing import Union

import cv2
import numpy as np
from data.iis_dataset import SegDataset
from scipy.io import loadmat


class SBDDataset(SegDataset):
    """Dataset with only one mask layer but with many objects in that layer
    (sometimes more than 10)
    returns:
        - image (H, W, 3), np.uint8 with max 255
        - mask (H, W, 1), int32 with max 1
    """

    def __init__(
        self,
        dataset_path: Union[str, Path],
        split: str = "train",
    ):
        super().__init__()
        assert split in {"train", "test"}
        split = "val" if split == "test" else "train"
        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / "img"
        self._insts_path = self.dataset_path / "inst"

        with open(
            self.dataset_path / f"{split}.txt", "r", encoding="ascii"
        ) as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]
        self.at_child_init_end()

    def get_sample(self, index: int) -> Tuple[np.ndarray, np.ndarray, Any]:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f"{image_name}.jpg")
        inst_info_path = str(self._insts_path / f"{image_name}.mat")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = loadmat(str(inst_info_path))["GTinst"][0][0][
            0
        ].astype(np.int32)[
            ..., None
        ]  # add channel dimension
        layers, info = instances_mask, None
        return image, layers, info
