#!/usr/bin/env python3
"""Evaluate Stage 0 D/A/B checkpoints on identical held-out datasets.

Examples
--------
python scripts/evaluate_stage0_checkpoints.py \
  --model D_proxy=/scratch/.../Stage0_D_proxy/net_g_latest.pth \
  --model A_proxy=/scratch/.../Stage0_A_proxy/net_g_latest.pth \
  --model B_proxy=/scratch/.../Stage0_B_proxy/net_g_latest.pth \
  --synthetic-test /scratch/.../stage0_synthetic_proxy/manifests/test.jsonl \
  --synthetic-unseen /scratch/.../stage0_synthetic_proxy/manifests/unseen_rain_snow.jsonl \
  --cdd11-test-root /scratch/.../datasets/CDD11/test \
  --output /scratch/.../stage0_evaluation_proxy.json

The A model was trained with severity_loss_weight=0.0. Its severity result is
still evaluated because A inherits the D severity head, but it must be
interpreted as a retention/forgetting test, not as new severity supervision.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from basicsr.archs import build_network
from basicsr.data import build_dataset


TYPES = ["low", "haze", "rain", "snow"]
CATEGORIES = [
    "clear", "low", "haze", "rain", "snow", "low_haze", "low_rain",
    "low_snow", "haze_rain", "haze_snow", "low_haze_rain",
    "low_haze_snow",
]


def load_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in ("params_ema", "params"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    cleaned = {}
    for key, value in payload.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {path}: missing={missing}, unexpected={unexpected}"
        )


def make_dataset(mode, path, cdd11_categories=None):
    if mode == "synthetic":
        options = {
            "name": "stage0_test",
            "type": "Stage0UnifiedDataset",
            "mode": "synthetic",
            "manifest": str(path),
            "degradation_types": TYPES,
            "crop_size": 224,
            "phase": "test",
        }
    else:
        options = {
            "name": "cdd11_test",
            "type": "Stage0UnifiedDataset",
            "mode": "cdd11",
            "dataroot_lq": str(path),
            "categories": cdd11_categories or CATEGORIES,
            "degradation_types": TYPES,
            "crop_size": 224,
            "phase": "test",
        }
    return build_dataset(options)


def make_loader(dataset, batch_size, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(model, loader, device, has_severity):
    model.eval()
    n_types = len(TYPES)
    type_tp = np.zeros(n_types, dtype=np.int64)
    type_fp = np.zeros(n_types, dtype=np.int64)
    type_fn = np.zeros(n_types, dtype=np.int64)
    type_exact = 0
    type_total = 0
    sev_correct = np.zeros(n_types, dtype=np.int64)
    sev_adjacent = np.zeros(n_types, dtype=np.int64)
    sev_abs_error = np.zeros(n_types, dtype=np.int64)
    sev_total = np.zeros(n_types, dtype=np.int64)
    confusion = np.zeros((n_types, 4, 4), dtype=np.int64)

    for batch in loader:
        image = batch["lq"].to(device, non_blocking=True)
        logits = model(image)
        present = logits.softmax(-1)[..., 1:].sum(-1) >= 0.5

        target_types = batch["type_labels"].to(device).bool()
        type_tp += (present & target_types).sum(0).cpu().numpy()
        type_fp += (present & ~target_types).sum(0).cpu().numpy()
        type_fn += (~present & target_types).sum(0).cpu().numpy()
        type_exact += (present == target_types).all(1).sum().item()
        type_total += target_types.shape[0]

        if has_severity:
            prediction = logits.argmax(-1)
            target = batch["severity_bins"].to(device)
            for index in range(n_types):
                pred_i = prediction[:, index]
                target_i = target[:, index]
                sev_correct[index] += (pred_i == target_i).sum().item()
                sev_adjacent[index] += ((pred_i - target_i).abs() <= 1).sum().item()
                sev_abs_error[index] += (pred_i - target_i).abs().sum().item()
                sev_total[index] += target_i.numel()
                for expected, predicted in zip(target_i.cpu().tolist(), pred_i.cpu().tolist()):
                    confusion[index, expected, predicted] += 1

    f1 = 2 * type_tp / np.maximum(2 * type_tp + type_fp + type_fn, 1)
    result = {
        "type": {
            "per_type_f1": dict(zip(TYPES, f1.round(6).tolist())),
            "macro_f1": float(f1.mean()),
            "exact_match": float(type_exact / max(type_total, 1)),
            "samples": int(type_total),
        }
    }
    if has_severity:
        valid = np.maximum(sev_total, 1)
        result["severity"] = {
            "per_type_accuracy": dict(zip(TYPES, (sev_correct / valid).round(6).tolist())),
            "per_type_mae": dict(zip(TYPES, (sev_abs_error / valid).round(6).tolist())),
            "per_type_adjacent_accuracy": dict(zip(TYPES, (sev_adjacent / valid).round(6).tolist())),
            "mean_accuracy": float((sev_correct / valid).mean()),
            "mean_mae": float((sev_abs_error / valid).mean()),
            "mean_adjacent_accuracy": float((sev_adjacent / valid).mean()),
            "confusion_matrices": confusion.tolist(),
            "samples": int(sev_total.max()),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT")
    parser.add_argument("--synthetic-test", required=True)
    parser.add_argument("--synthetic-unseen", required=True)
    parser.add_argument("--cdd11-test-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    models = {}
    for item in args.model:
        if "=" not in item:
            parser.error(f"--model must be NAME=CHECKPOINT, got {item}")
        name, checkpoint = item.split("=", 1)
        models[name] = checkpoint

    datasets = {
        "cdd11_test": (make_dataset("cdd11", args.cdd11_test_root), False),
        "synthetic_test": (make_dataset("synthetic", args.synthetic_test), True),
        "synthetic_unseen_rain_snow": (make_dataset("synthetic", args.synthetic_unseen), True),
    }
    loaders = {
        name: make_loader(dataset, args.batch_size, args.workers)
        for name, (dataset, _) in datasets.items()
    }

    device = torch.device(args.device)
    results = {
        "models": models,
        "datasets": {name: len(dataset) for name, (dataset, _) in datasets.items()},
        "degradation_types": TYPES,
        "results": {},
    }
    for name, checkpoint in models.items():
        network = build_network({
            "type": "OrdinalDegradationExtractor",
            "num_types": 4,
            "num_bins": 4,
            "widths": [64, 128, 256, 384],
        }).to(device)
        load_checkpoint(network, checkpoint)
        results["results"][name] = {}
        for dataset_name, (dataset, has_severity) in datasets.items():
            results["results"][name][dataset_name] = evaluate(
                network, loaders[dataset_name], device, has_severity
            )
        del network
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
