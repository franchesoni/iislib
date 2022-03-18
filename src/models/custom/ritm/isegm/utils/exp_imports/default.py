import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from models.custom.ritm.isegm.data.datasets import *
from models.custom.ritm.isegm.model.losses import *
from models.custom.ritm.isegm.data.transforms import *
from models.custom.ritm.isegm.engine.trainer import ISTrainer
from models.custom.ritm.isegm.model.metrics import AdaptiveIoU
from models.custom.ritm.isegm.data.points_sampler import MultiPointSampler
from models.custom.ritm.isegm.utils.log import logger
from models.custom.ritm.isegm.model import initializer

from models.custom.ritm.isegm.model.is_hrnet_model import HRNetModel
from models.custom.ritm.isegm.model.is_deeplab_model import DeeplabModel