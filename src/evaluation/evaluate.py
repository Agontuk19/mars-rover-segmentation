"""Final evaluation of a trained model on the official test splits."""
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.metrics import CLASS_NAMES
from src.models.unet import UNet
from src.training.train import evaluate


def load_best_model(ckpt_path, device, model=None, base=32, num_classes=4):
    """Load weights saved by `fit`. If no model is given, build the from-scratch U-Net."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if model is None:
        model = UNet(in_channels=3, num_classes=num_classes, base=base)
    model = model.to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded epoch {ckpt['epoch']} (val mIoU at save time: {ckpt['val_miou']:.4f})")
    return model


def evaluate_splits(model, loaders, device, ignore_index=255, use_amp=False, return_confusion=False):
    """Evaluate on each loader in `loaders` (dict: name -> DataLoader).

    Returns a DataFrame with one row per split plus a 'mean' row.
    With return_confusion=True returns (DataFrame, {split: confusion matrix}).
    """
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    rows, confusions = [], {}

    for split, loader in loaders.items():
        loss, res = evaluate(model, loader, criterion, device,
                             ignore_index=ignore_index, use_amp=use_amp, desc=split)
        confusions[split] = res["confusion"]
        rows.append({                                                                   # Stores evaluation metrics
            "split": split, "loss": loss, "miou": res["miou"], "pixel_acc": res["pixel_acc"],
            **{f"iou_{CLASS_NAMES[k]}": float(v) for k, v in enumerate(res["iou"])},
        })

    df = pd.DataFrame(rows)
    mean_row = {"split": "mean", **df.drop(columns="split").mean().to_dict()}           # Calculates mean
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    return (df, confusions) if return_confusion else df