"""Post-hoc class-bias sweep: shift one class's logit and measure the effect (no retraining)."""
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.evaluation.metrics import CLASS_NAMES, ConfusionMatrix


@torch.no_grad()
def sweep_class_bias(model, loader, device, class_id=3, biases=(0.0,),
                     num_classes=4, ignore_index=255, use_amp=False):
    """Evaluate once per bias by adding it to `class_id`'s logit before argmax.
    The forward pass runs once per batch and is shared by all biases.
    Returns a DataFrame: one row per bias with mIoU, per-class IoU, and
    precision/recall of `class_id`."""
    model.eval()
    cms = {b: ConfusionMatrix(num_classes, ignore_index) for b in biases}

    for images, labels in tqdm(loader, desc="bias sweep", leave=False):
        images = images.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        logits = logits.float()
        for b in biases:
            adjusted = logits.clone()
            adjusted[:, class_id] += b
            cms[b].update(adjusted.argmax(dim=1), labels)

    rows = []
    for b, cm in cms.items():
        res = cm.compute()
        conf = res["confusion"]
        tp = conf[class_id, class_id]
        rows.append({
            "bias": b,
            "miou": res["miou"],
            **{f"iou_{CLASS_NAMES[k]}": float(v) for k, v in enumerate(res["iou"])},
            "precision": float(tp / max(conf[:, class_id].sum(), 1)),
            "recall": float(tp / max(conf[class_id, :].sum(), 1)),
        })
    return pd.DataFrame(rows)