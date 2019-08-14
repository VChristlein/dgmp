import numpy as np
from PIL import Image
from skimage import feature
from torchvision import transforms

import config


class PatchCrop(object):

    def __init__(self, patch_size, resize_to=256):
        self.patch_size = patch_size

    def __call__(self, sample):
        img = np.asarray(sample)
        height, width = img.shape[0], img.shape[1]
        patches = []
        resize = transforms.Resize(256)
        for i in range(0, height - self.patch_size, self.patch_size):
            for j in range(0, width - self.patch_size, self.patch_size):
                patch = img[i: i + self.patch_size, j: j + self.patch_size, :]
                # normalize img to [0, 1] range and sort out patches
                patch_normalized = patch / 255.0
                patch_normalized = patch_normalized.mean(axis=2)
                num_features = feature.canny(patch_normalized, sigma=config.CANNY_SIGMA).sum()
                if num_features > config.THRESHOLD_FEATURES:
                    patches.append(patch)
                # patches.append(patch)
        pil_patches = [resize(Image.fromarray(np.uint8(p))) for p in patches]
        return pil_patches




