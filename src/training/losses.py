"""Build a loss from an experiment config dict. 255 is always ignored."""
import torch
import torch.nn as nn

IGNORE_INDEX = 255


def build_loss(cfg, device):
    kind = cfg.get("loss", "ce")

    if kind == "ce":
        return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    if kind == "ce_weighted":
        weights = torch.tensor(cfg["class_weights"], dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)

    raise ValueError(f"Unknown loss type: {kind}")