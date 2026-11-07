"""Segmentation metrics from an accumulated confusion matrix."""
import numpy as np
import torch

CLASS_NAMES = {0: "Soil", 1: "Bedrock", 2: "Sand", 3: "Big Rock"}


class ConfusionMatrix:
    """Rows = true class, columns = predicted class. Pixels with target == ignore_index are skipped."""

    def __init__(self, num_classes=4, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def reset(self):
        self.mat.zero_()

    @torch.no_grad()
    def update(self, preds, targets):
        """preds, targets: integer tensors of identical shape (any device)."""
        preds = preds.detach().cpu().reshape(-1)            # Move tensors to CPU and flatten them into 1D
        targets = targets.detach().cpu().reshape(-1)

        valid = targets != self.ignore_index                # Remove ignored pixels
        preds, targets = preds[valid], targets[valid]

        idx = targets * self.num_classes + preds            # Convert predictions into indices
        counts = torch.bincount(idx, minlength=self.num_classes ** 2)       # Count occurrences
        self.mat += counts.reshape(self.num_classes, self.num_classes)      # Reconstruct the confusion matrix

    def compute(self):
        mat = self.mat.double()
        tp = torch.diag(mat)      # True Positives in diagonal
        gt = mat.sum(dim=1)       # true pixels per class
        pred = mat.sum(dim=0)     # predicted pixels per class
        union = gt + pred - tp    # Intersection over Union

        # IoU means better segmentation performance
        iou = torch.where(
            union > 0,
            tp / union,                         # union > 0, calculate IoU
            torch.full_like(tp, float("nan"))   # return NaN
        )
        # Recall measures how many actual class pixels were correctly identified.
        recall = torch.where(
            gt > 0, tp / gt,
            torch.full_like(tp, float("nan")))
    
        return {
            "iou": iou.numpy(),
            "recall": recall.numpy(),
            "miou": float(np.nanmean(iou.numpy())),     # Mean Intersection over Union
            # Pixel Accuracy measures the proportion of correctly classified pixels across all classes.
            "pixel_acc": float(tp.sum() / mat.sum()) if mat.sum() > 0 else float("nan"),
            "pixels_evaluated": int(mat.sum()),         # Total number of evaluated pixels
        }


def format_report(result):
    """Readable one-block summary of compute() output."""
    lines = [f"mIoU: {result['miou']:.4f} | pixel acc: {result['pixel_acc']:.4f}"]
    for k, name in CLASS_NAMES.items():
        lines.append(f"  {name:<9} IoU {result['iou'][k]:.4f} | recall {result['recall'][k]:.4f}")
    return "\n".join(lines)