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

