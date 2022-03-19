from pathlib import Path
import pickle
import os
import numpy as np
import cv2
from copy import deepcopy

from data.iis_dataset import SegDataset

          
import matplotlib.pyplot as plt

class GrabCutDataset(SegDataset):
    '''Database whose masks are represented with values 0: background, 128: border, 255: foreground.'''
    def __init__(
        self,
        dataset_path,
        images_dir_name='data_GT',
        masks_dir_name='boundary_GT',
    ):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name
        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.*')}
        self.check_sample()

    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)[..., None]
        instances_mask[instances_mask >= 128] = 1
        layers, info = instances_mask, None
        return image, instances_mask, info
