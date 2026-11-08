"""Joint image/label augmentation for segmentation (training data only)."""
import random

import numpy as np
from PIL import Image, ImageEnhance


class SegAugment:
    """Horizontal flip, random zoom-in crop, and brightness/contrast jitter.

    Geometry is applied identically to image and label (label uses nearest-neighbor,
    so class IDs and 255 stay intact). Color jitter touches the image only.
    No vertical flip: sky is up and ground is down in rover images.
    Uses Python's `random`, which PyTorch seeds separately in each DataLoader worker.
    """

    def __init__(self, hflip=0.5, crop_frac=(0.7, 1.0), brightness=0.25, contrast=0.25):
        self.hflip = hflip
        self.crop_frac = crop_frac
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, label):
        """image: PIL RGB (S, S). label: uint8 array (S, S). Returns the same types and shapes."""
        size = image.size[0]

        if random.random() < self.hflip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.ascontiguousarray(label[:, ::-1])

        side = int(size * random.uniform(*self.crop_frac))
        if side < size:
            x0 = random.randint(0, size - side)
            y0 = random.randint(0, size - side)
            image = image.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.BILINEAR)
            crop = Image.fromarray(label[y0:y0 + side, x0:x0 + side])
            label = np.array(crop.resize((size, size), Image.NEAREST), dtype=np.uint8)

        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(1 - self.brightness, 1 + self.brightness))
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(1 - self.contrast, 1 + self.contrast))
        return image, label


def build_train_transform(cfg):
    """Return the training transform named by cfg['augment'], or None for no augmentation."""
    name = cfg.get("augment")
    if not name:
        return None
    if name == "basic":
        return SegAugment()
    raise ValueError(f"Unknown augment type: {name}")