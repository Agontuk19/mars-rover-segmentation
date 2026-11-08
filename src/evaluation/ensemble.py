"""Evaluate one model or an ensemble, optionally with horizontal-flip test-time augmentation."""
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.evaluation.metrics import CLASS_NAMES, ConfusionMatrix

BIG_ROCK = 3


@torch.no_grad()
def predict_probs(models, images, flip_tta=False, use_amp=False):
    """Average class probabilities over models, and over a horizontal flip when flip_tta=True."""
    total, count = None, 0
    for model in models:
        model.eval()
        for flipped in ([False, True] if flip_tta else [False]):
            view = torch.flip(images, dims=[3]) if flipped else images
            with torch.autocast(device_type=images.device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(view)
            probs = logits.float().softmax(dim=1)
            if flipped:
                probs = torch.flip(probs, dims=[3])      # flip the prediction back
            total = probs if total is None else total + probs
            count += 1
    return total / count


@torch.no_grad()
def evaluate_ensemble(models, loader, device, flip_tta=False, use_amp=False,
                      num_classes=4, ignore_index=255, desc="eval"):
    """Confusion-matrix metrics for the (ensemble of) model(s) on one loader."""
    cm = ConfusionMatrix(num_classes, ignore_index)
    for images, labels in tqdm(loader, desc=desc, leave=False):
        probs = predict_probs(models, images.to(device), flip_tta, use_amp)
        cm.update(probs.argmax(dim=1), labels)
    return cm.compute()


def summarize(results_by_split, label):
    """One row per split plus a mean row, including Big Rock precision and recall."""
    rows = []
    for split, res in results_by_split.items():
        conf = res["confusion"]
        tp = conf[BIG_ROCK, BIG_ROCK]
        rows.append({
            "setup": label, "split": split, "miou": res["miou"], "pixel_acc": res["pixel_acc"],
            **{f"iou_{CLASS_NAMES[k]}": float(v) for k, v in enumerate(res["iou"])},
            "bigrock_precision": float(tp / max(conf[:, BIG_ROCK].sum(), 1)),
            "bigrock_recall": float(tp / max(conf[BIG_ROCK, :].sum(), 1)),
        })
    df = pd.DataFrame(rows)
    mean = df.drop(columns=["setup", "split"]).mean().to_dict()
    return pd.concat([df, pd.DataFrame([{"setup": label, "split": "mean", **mean}])], ignore_index=True)