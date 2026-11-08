"""Training and evaluation loops for semantic segmentation.

Features: mixed precision (CUDA only), per-epoch resumable checkpoints,
best-model tracking by validation mIoU, and a JSON training history.
"""
import json
import time
from itertools import islice
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.evaluation.metrics import ConfusionMatrix

# Helpers
# --------------------------------------------------------------------------- 
def _limit_batches(loader, max_batches):
    if max_batches:                     # For trail batches
        return islice(loader, max_batches), min(len(loader), max_batches)
    return loader, len(loader)          # each iteration details, total iterations (Used for tqdm process show)


def _save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, history, best_miou):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "history": history,
        "best_miou": best_miou,
    }, path)


def _load_checkpoint(path, model, optimizer, scaler, scheduler, device):
    """Restore training state. Returns (history, best_miou, last_finished_epoch)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["history"], ckpt["best_miou"], ckpt["epoch"]


# One epoch of training / evaluation
# --------------------------------------------------------------------------- 
def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, max_batches=None, desc="train"):
    """One pass over `loader`. Returns (mean loss, seconds per batch)."""
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    batches, total = _limit_batches(loader, max_batches)

    total_loss, n_batches = 0.0, 0      # Tracking variables
    start = time.time()

    for images, labels in tqdm(batches, total=total, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Forward Propagation
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        loss = criterion(logits.float(), labels)   # loss computed in fp32 for stability

        # Invalid loss check
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss. Check the data, learning rate, or batch contents.")

        # Backpropagation
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), (time.time() - start) / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=4, ignore_index=255,
             use_amp=False, max_batches=None, desc="eval"):
    """Evaluate on `loader`. Returns (mean loss, metrics dict from ConfusionMatrix.compute())."""
    model.eval()
    cm = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    batches, total = _limit_batches(loader, max_batches)

    total_loss, n_batches = 0.0, 0
    for images, labels in tqdm(batches, total=total, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward Pass without gradients
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
        logits = logits.float()

        # Validation loss
        total_loss += criterion(logits, labels).item()
        n_batches += 1
        cm.update(logits.argmax(dim=1), labels)     # Logits -> Class Prediction

    return total_loss / max(n_batches, 1), cm.compute()


# Full training run
# --------------------------------------------------------------------------- 
def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs, out_dir,
        scheduler=None, use_amp=True, resume=True,
        max_train_batches=None, max_val_batches=None):
    """Train for `epochs` epochs.

    Writes to `out_dir`:
        last.pt       full state after every epoch (used for resuming)
        best.pt       model weights with the best validation mIoU
        history.json  per-epoch losses and metrics
    Returns the history list.
    """
    # Output files 
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path, history_path = out_dir / "last.pt", out_dir / "best.pt", out_dir / "history.json"

    # Configure AMP
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_miou, start_epoch = [], -1.0, 1
    # Resumes previous trainings
    if resume and last_path.exists():
        history, best_miou, done = _load_checkpoint(last_path, model, optimizer, scaler, scheduler, device)
        start_epoch = done + 1
        print(f"Resumed after epoch {done} (best val mIoU so far: {best_miou:.4f})")

    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_loss, sec_per_batch = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler=scaler,
            max_batches=max_train_batches, desc=f"epoch {epoch} train")
        
        # Validate
        val_loss, val = evaluate(
            model, val_loader, criterion, device, use_amp=use_amp,
            max_batches=max_val_batches, desc=f"epoch {epoch} val")

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Records training stats
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": val["miou"],
            "val_pixel_acc": val["pixel_acc"],
            **{f"val_iou_{k}": float(v) for k, v in enumerate(val["iou"])},
            "lr": optimizer.param_groups[0]["lr"],
            "sec_per_batch": sec_per_batch,
        })

        # Saves best model
        if val["miou"] > best_miou:
            best_miou = val["miou"]
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_miou": best_miou}, best_path)

        # Saves last epoch
        _save_checkpoint(last_path, model, optimizer, scaler, scheduler, epoch, history, best_miou)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Print Progress
        ious = [round(float(v), 3) for v in val["iou"]]
        print(f"Epoch {epoch:>2}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"mIoU {val['miou']:.4f} | IoU {ious} | {sec_per_batch:.2f}s/batch")

    return history