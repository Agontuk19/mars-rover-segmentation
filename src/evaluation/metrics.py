"""Segmentation metrics from an accumulated confusion matrix (ignore label excluded)."""
import numpy as np
import torch

CLASS_NAMES = {0: "Soil", 1: "Bedrock", 2: "Sand", 3: "Big Rock"}


def scores_from_confusion(mat):
    """Compute metrics from a (C, C) confusion matrix (rows = true, columns = predicted)."""
    mat = np.asarray(mat, dtype=np.float64)
    tp = np.diag(mat)
    gt = mat.sum(axis=1)          # true pixels per class
    pred = mat.sum(axis=0)        # predicted pixels per class
    union = gt + pred - tp
    total = mat.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        recall = np.where(gt > 0, tp / gt, np.nan)

    return {
        "iou": iou,
        "recall": recall,
        "miou": float(np.nanmean(iou)),
        "pixel_acc": float(tp.sum() / total) if total > 0 else float("nan"),
        "pixels_evaluated": int(total),
        "confusion": mat.astype(np.int64),
    }


class ConfusionMatrix:
    """Accumulates a confusion matrix over batches. Pixels with target == ignore_index are skipped."""

    def __init__(self, num_classes=4, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def reset(self):
        self.mat.zero_()

    @torch.no_grad()
    def update(self, preds, targets):
        """preds, targets: integer tensors of identical shape (any device)."""
        preds = preds.detach().cpu().reshape(-1)
        targets = targets.detach().cpu().reshape(-1)
        valid = targets != self.ignore_index                            # Excludes ignored pixels
        preds, targets = preds[valid], targets[valid]
        idx = targets * self.num_classes + preds
        counts = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.mat += counts.reshape(self.num_classes, self.num_classes)  # 16 -> 4x4

    # Confusion Matrix to NumPy array
    def compute(self):
        return scores_from_confusion(self.mat.numpy())


def format_report(result):
    """Readable summary of a metrics dict."""
    lines = [f"mIoU: {result['miou']:.4f} | pixel acc: {result['pixel_acc']:.4f}"]
    for k, name in CLASS_NAMES.items():
        lines.append(f"  {name:<9} IoU {result['iou'][k]:.4f} | recall {result['recall'][k]:.4f}")
    return "\n".join(lines)