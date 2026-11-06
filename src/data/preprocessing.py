"""AI4MARS preprocessing: safe decoding, rover/range exclusions, resizing, Dataset."""
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IGNORE_INDEX = 255
CLASS_NAMES = {0: "Soil", 1: "Bedrock", 2: "Sand", 3: "Big Rock"}
VALID_LABELS = {0, 1, 2, 3, 255}
IMG_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_raw_image(image_data):
    """Decode a decode=False Hugging Face image dict into a PIL image, or None."""
    if image_data is None:
        return None
    try:
        if image_data.get("bytes") is not None:
            img = Image.open(BytesIO(image_data["bytes"]))
        else:
            img = Image.open(image_data["path"])
        img.load()
        return img
    except Exception:
        return None


def to_2d_array(pil_img):
    """Convert a PIL mask to a 2D NumPy array (drops a channel axis if present)."""
    if pil_img is None:
        return None
    arr = np.array(pil_img)
    return arr[..., 0] if arr.ndim == 3 else arr


def apply_exclusions(label, rover=None, range_=None):
    """Set rover-body and >30 m pixels to 255 on a copy of `label`.
    Returns (processed_label, info). Skipped exclusions are recorded in `info`."""
    processed = label.astype(np.uint8).copy()
    already_ignored = label == IGNORE_INDEX
    info = {
        "rover_applied": rover is not None,
        "range_applied": range_ is not None,
        "rover_new_ignored": 0,
        "range_new_ignored": 0,
        "overlap_new_ignored": 0,
    }
    rover_region = range_region = None

    if rover is not None:
        if rover.shape != label.shape:
            raise ValueError(f"rover shape {rover.shape} != label shape {label.shape}")
        rover_region = rover > 0
        info["rover_new_ignored"] = int((rover_region & ~already_ignored).sum())
        processed[rover_region] = IGNORE_INDEX

    if range_ is not None:
        if range_.shape != label.shape:
            raise ValueError(f"range shape {range_.shape} != label shape {label.shape}")
        range_region = range_ > 0
        info["range_new_ignored"] = int((range_region & ~already_ignored).sum())
        processed[range_region] = IGNORE_INDEX

    if rover_region is not None and range_region is not None:
        info["overlap_new_ignored"] = int((rover_region & range_region & ~already_ignored).sum())
    return processed, info


def resize_image(image, size=IMG_SIZE):
    """Bilinear resize for images."""
    return image.resize((size, size), resample=Image.BILINEAR)


def resize_label(label, size=IMG_SIZE):
    """Nearest-neighbor resize for label masks (never invents class values)."""
    pil_label = Image.fromarray(label.astype(np.uint8), mode="L")
    return np.array(pil_label.resize((size, size), resample=Image.NEAREST), dtype=np.uint8)


class Ai4MarsSegmentation(Dataset):
    """Returns (image float32 (3,H,W) normalized, label int64 (H,W) in {0,1,2,3,255})."""

    def __init__(self, raw_split, indices, img_size=IMG_SIZE, normalize=True):
        self.data = raw_split
        self.indices = [int(i) for i in indices]
        self.img_size = img_size
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = self.indices[k]
        row = self.data[idx]

        image = decode_raw_image(row["image"])
        label = to_2d_array(decode_raw_image(row["label_mask"]))
        rover = to_2d_array(decode_raw_image(row["rover_mask"]))
        range_ = to_2d_array(decode_raw_image(row["range_mask"]))

        if image is None:
            raise ValueError(f"Row {idx}: image missing or corrupted")
        if label is None:
            raise ValueError(f"Row {idx}: label mask missing")

        label, _ = apply_exclusions(label, rover, range_)
        label = resize_label(label, self.img_size)

        image = resize_image(image.convert("RGB"), self.img_size)
        image = np.asarray(image, dtype=np.float32) / 255.0
        if self.normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD

        image_t = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return image_t, torch.from_numpy(label.astype(np.int64))