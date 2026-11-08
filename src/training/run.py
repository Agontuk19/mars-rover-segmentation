"""Run one experiment end to end from a config dict."""
import json
from pathlib import Path

import torch

from src.data.dataset import build_test_loaders, build_train_val_loaders
from src.evaluation.evaluate import evaluate_splits, load_best_model
from src.models.factory import build_model
from src.training.losses import build_loss
from src.training.train import fit
from src.utils.common import seed_everything
from src.data.transforms import build_train_transform


def run_experiment(cfg, raw_datasets, splits, device, out_root):
    """Train with cfg, evaluate best.pt on the three test splits, save everything under out_root/<name>/.

    Required keys: name, model, epochs.
    Optional keys: loss, lr, weight_decay, img_size, batch_size, num_workers, seed,
                   encoder, encoder_weights, base, class_weights,
                   max_train_batches, max_val_batches (for quick smoke runs).
    """
    out_dir = Path(out_root) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    seed_everything(cfg.get("seed", 42))
    img_size = cfg.get("img_size", 512)
    batch_size = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 2)
    pin = device.type == "cuda"

    train_loader, val_loader = build_train_val_loaders(
        raw_datasets, splits, img_size, batch_size, num_workers, pin_memory=pin,
        train_transform=build_train_transform(cfg))

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg['name']}] {cfg['model']} | {n_params / 1e6:.1f}M parameters | loss: {cfg.get('loss', 'ce')}")

    history = fit(model, train_loader, val_loader, optimizer, criterion, device,
                  epochs=cfg["epochs"], out_dir=out_dir, scheduler=scheduler, resume=True,
                  max_train_batches=cfg.get("max_train_batches"),
                  max_val_batches=cfg.get("max_val_batches"))

    model = load_best_model(out_dir / "best.pt", device, model=model)
    test_loaders = build_test_loaders(raw_datasets, img_size, batch_size, num_workers, pin_memory=pin)
    results = evaluate_splits(model, test_loaders, device, use_amp=pin)
    results.to_csv(out_dir / "test_results.csv", index=False)
    print(results.round(4).to_string(index=False))
    return history, results