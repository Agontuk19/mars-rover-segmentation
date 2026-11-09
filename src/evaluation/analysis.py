"""Per-image confusion matrices, bootstrap confidence intervals, and cross-fitted Big Rock bias."""
import numpy as np
import torch
from tqdm.auto import tqdm

from src.evaluation.ensemble import predict_probs
from src.evaluation.metrics import scores_from_confusion


@torch.no_grad()
def per_image_confusions(models, loader, device, biases=(0.0,), big_rock=3, num_classes=4,
                         ignore_index=255, use_amp=False, desc="per-image"):
    """Array (N, n_biases, C, C): the confusion matrix of every image for each bias.
    A bias is added to Big Rock's log-probability before the argmax (0.0 = plain argmax)."""
    out = []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        probs = predict_probs(models, images.to(device), flip_tta=False, use_amp=use_amp)
        logp = probs.clamp_min(1e-8).log()
        labels = labels.to(device)

        preds_per_bias = []
        for b in biases:
            adjusted = logp.clone()
            adjusted[:, big_rock] += b
            preds_per_bias.append(adjusted.argmax(dim=1))

        for i in range(images.shape[0]):
            valid = labels[i] != ignore_index
            target = labels[i][valid]
            mats = [torch.bincount(target * num_classes + preds[i][valid],
                                   minlength=num_classes ** 2).reshape(num_classes, num_classes)
                    for preds in preds_per_bias]
            out.append(torch.stack(mats).cpu())
    return torch.stack(out).numpy()


def mean_miou(conf_by_split, image_idx, bias_idx=0):
    """Mean over the test splits of the mIoU computed from the chosen images' pooled confusion."""
    vals = [scores_from_confusion(conf[image_idx, bias_idx].sum(axis=0))["miou"]
            for conf in conf_by_split.values()]
    return float(np.mean(vals))


def paired_bootstrap(conf_a, conf_b, n_boot=1000, seed=0, bias_idx=0):
    """Resample test images with replacement. Returns (mean difference B - A, (2.5%, 97.5%), P(B > A))."""
    rng = np.random.default_rng(seed)
    n = next(iter(conf_a.values())).shape[0]
    diffs = np.array([
        mean_miou(conf_b, idx, bias_idx) - mean_miou(conf_a, idx, bias_idx)
        for idx in (rng.integers(0, n, n) for _ in range(n_boot))
    ])
    return float(diffs.mean()), tuple(np.percentile(diffs, [2.5, 97.5])), float((diffs > 0).mean())


def crossfit_bias(conf_by_split, biases, n_repeats=20, seed=0):
    """Pick the best bias on a random half of the images, score it on the other half, repeat.
    Returns (mean tuned score, mean untuned score on the same halves, list of chosen biases).
    `biases` must contain 0.0."""
    rng = np.random.default_rng(seed)
    zero = list(biases).index(0.0)
    n = next(iter(conf_by_split.values())).shape[0]
    tuned, untuned, chosen = [], [], []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        halves = (perm[: n // 2], perm[n // 2:])
        for fit_idx, eval_idx in (halves, halves[::-1]):
            best = int(np.argmax([mean_miou(conf_by_split, fit_idx, k) for k in range(len(biases))]))
            tuned.append(mean_miou(conf_by_split, eval_idx, best))
            untuned.append(mean_miou(conf_by_split, eval_idx, zero))
            chosen.append(biases[best])
    return float(np.mean(tuned)), float(np.mean(untuned)), chosen