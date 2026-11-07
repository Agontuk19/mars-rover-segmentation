"""Training and evaluation loops for semantic segmentation."""
import time
from itertools import islice
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.evaluation.metrics import ConfusionMatrix


def train_one_epoch(model, loader, optimizer, criterion, device, max_batches=None, desc="train"):
    """One pass over `loader`. Returns mean loss and seconds per batch."""
    model.train()
    total_loss, n_batches = 0.0, 0
    start = time.time()

    batches = islice(loader, max_batches) if max_batches else loader
    total = min(len(loader), max_batches) if max_batches else len(loader) if hasattr(loader, "__len__") else None

    for images, labels in tqdm(batches, total=total, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss. Check the data, learning rate, or batch contents.")
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    elapsed = time.time() - start
    return total_loss / max(n_batches, 1), elapsed / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=4, ignore_index=255, max_batches=None, desc="eval"):
    """Evaluate on `loader`. Returns (mean loss, metrics dict from ConfusionMatrix.compute())."""
    model.eval()
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    total_loss, n_batches = 0.0, 0

    batches = islice(loader, max_batches) if max_batches else loader
    total = min(len(loader), max_batches) if max_batches else len(loader) if hasattr(loader, "__len__") else None

    for images, labels in tqdm(batches, total=total, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        n_batches += 1
        cm.update(logits.argmax(dim=1), labels)

    return total_loss / max(n_batches, 1), cm.compute()


def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs,
        checkpoint_path, max_train_batches=None, max_val_batches=None):
    """Train for `epochs`, keep the checkpoint with the best validation mIoU.
    Returns a list of per-epoch dicts (history)."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history, best_miou = [], -1.0
    for epoch in range(1, epochs + 1):
        train_loss, sec_per_batch = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            max_batches=max_train_batches, desc=f"epoch {epoch} train")
        val_loss, val_res = evaluate(
            model, val_loader, criterion, device,
            max_batches=max_val_batches, desc=f"epoch {epoch} val")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val_res["miou"],
            "val_pixel_acc": val_res["pixel_acc"],
            **{f"val_iou_{k}": float(v) for k, v in enumerate(val_res["iou"])},
            "sec_per_batch": sec_per_batch,
        }
        history.append(record)

        if val_res["miou"] > best_miou:
            best_miou = val_res["miou"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_miou": best_miou},
                       checkpoint_path)

        print(f"Epoch {epoch:>2} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val mIoU {val_res['miou']:.4f} | {sec_per_batch:.2f}s/batch")

    return history