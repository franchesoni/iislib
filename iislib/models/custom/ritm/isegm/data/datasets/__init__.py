from models.custom.ritm.isegm.data.compose import (
    ComposeDataset,
    ProportionalComposeDataset,
)

from .ade20k import ADE20kDataset
from .berkeley import BerkeleyDataset
from .coco import CocoDataset
from .coco_lvis import CocoLvisDataset
from .davis import DavisDataset
from .grabcut import GrabCutDataset
from .images_dir import ImagesDirDataset
from .lvis import LvisDataset
from .openimages import OpenImagesDataset
from .pascalvoc import PascalVocDataset
from .sbd import SBDDataset, SBDEvaluationDataset
