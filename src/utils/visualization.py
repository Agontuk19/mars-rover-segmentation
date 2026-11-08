"""Plotting helpers: label colors, confusion matrices, prediction panels."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from src.data.preprocessing import IGNORE_INDEX, IMAGENET_MEAN, IMAGENET_STD
from src.evaluation.metrics import CLASS_NAMES

CLASS_COLORS = ["#c2a15f", "#6b6b6b", "#f2e394", "#d1495b"]   # soil, bedrock, sand, big rock
IGNORE_COLOR = "#1b1b1b"


def _hex_to_rgb(color):
    return [int(color[i:i + 2], 16) for i in (1, 3, 5)]


_PALETTE = np.array([_hex_to_rgb(c) for c in CLASS_COLORS + [IGNORE_COLOR]], dtype=np.uint8)


def colorize(label):
    """(H, W) label array with values {0,1,2,3,255} -> (H, W, 3) uint8 RGB image."""
    idx = np.where(label == IGNORE_INDEX, len(CLASS_COLORS), label)
    return _PALETTE[idx]


def denormalize(image_tensor):
    """Normalized (3, H, W) tensor -> (H, W, 3) float array in [0, 1] for display."""
    arr = image_tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(arr * IMAGENET_STD + IMAGENET_MEAN, 0.0, 1.0)


def plot_confusion_matrix(conf, ax, title=""):
    """Row-normalized confusion matrix (each row sums to 1: 'of the true X pixels, where did they go')."""
    conf = np.asarray(conf, dtype=np.float64)
    norm = conf / np.clip(conf.sum(axis=1, keepdims=True), 1, None)
    names = list(CLASS_NAMES.values())

    ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)


def find_indices_with_class(dataset, class_id, n=3, max_scan=150, min_pixels=200):
    """First `n` dataset indices whose label contains at least `min_pixels` of `class_id`."""
    found = []
    for i in range(min(len(dataset), max_scan)):
        _, label = dataset[i]
        if int((label == class_id).sum()) >= min_pixels:
            found.append(i)
            if len(found) == n:
                break
    return found


def show_predictions(model, dataset, indices, device, use_amp=False):
    """One row per sample: image | ground truth | prediction | error map."""
    model.eval()
    fig, axes = plt.subplots(len(indices), 4, figsize=(16, 4 * len(indices)), squeeze=False)

    for row, idx in enumerate(indices):
        image_t, label_t = dataset[idx]
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(image_t.unsqueeze(0).to(device))
        pred = logits.float().argmax(dim=1)[0].cpu().numpy()
        label = label_t.numpy()

        ignored = label == IGNORE_INDEX
        wrong = (pred != label) & ~ignored
        error_img = np.full(label.shape + (3,), 215, dtype=np.uint8)   # correct = light gray
        error_img[wrong] = (200, 30, 30)                               # wrong = red
        error_img[ignored] = _hex_to_rgb(IGNORE_COLOR)                 # ignored = dark

        panels = [
            (denormalize(image_t), "Image"),
            (colorize(label), "Ground truth"),
            (colorize(np.where(ignored, IGNORE_INDEX, pred)), "Prediction"),
            (error_img, f"Errors ({wrong.sum() / max((~ignored).sum(), 1) * 100:.1f}% of labeled px)"),
        ]
        for col, (img, title) in enumerate(panels):
            axes[row, col].imshow(img, interpolation="nearest")
            axes[row, col].set_title(f"#{idx} {title}" if col == 0 else title)
            axes[row, col].axis("off")

    handles = [Patch(color=c, label=n) for c, n in zip(CLASS_COLORS + [IGNORE_COLOR],
                                                        list(CLASS_NAMES.values()) + ["Ignore"])]
    fig.legend(handles=handles, loc="lower center", ncol=5)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()