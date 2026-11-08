"""Loading AI4MARS from Hugging Face and building DataLoaders."""
import json

from datasets import Image as HFImage
from datasets import load_dataset
from torch.utils.data import DataLoader

from src.data.preprocessing import IMG_SIZE, Ai4MarsSegmentation

HF_DATASET_NAME = "hassanjbara/AI4MARS"
IMAGE_COLUMNS = ["image", "label_mask", "rover_mask", "range_mask"]
TEST_SPLITS = ["test_min1", "test_min2", "test_min3"]


def load_raw_datasets(name=HF_DATASET_NAME):
    """Load every split with automatic image decoding disabled (we decode ourselves)."""
    dataset = load_dataset(name)
    raw = {}
    for split, data in dataset.items():
        for column in IMAGE_COLUMNS:
            data = data.cast_column(column, HFImage(decode=False))
        raw[split] = data
    return raw


def load_splits(path):
    """Read the saved train/val row indices (data/splits.json)."""
    with open(path) as f:
        return json.load(f)


def build_train_val_loaders(raw_datasets, splits, img_size=IMG_SIZE, batch_size=8,
                            num_workers=2, pin_memory=False):
    train_ds = Ai4MarsSegmentation(raw_datasets["train"], splits["train"], img_size=img_size)
    val_ds = Ai4MarsSegmentation(raw_datasets["train"], splits["val"], img_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def build_test_loaders(raw_datasets, img_size=IMG_SIZE, batch_size=8,
                       num_workers=2, pin_memory=False):
    """One DataLoader per official test split (all rows, no filtering)."""
    loaders = {}
    for split in TEST_SPLITS:
        ds = Ai4MarsSegmentation(raw_datasets[split], range(len(raw_datasets[split])), img_size=img_size)
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=pin_memory)
    return loaders