"""Generate reproducible synthetic severity-labelled Stage 0 scenes.

This is intentionally an auditable baseline generator. Every output record
stores the chosen bins and continuous parameters so severity bins can be
inspected and revised without changing the model code.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _darken(image, level):
    gamma = [1.0, 1.35, 1.9, 2.6][level]
    return np.clip(image ** gamma, 0, 1)


def _haze(image, level):
    alpha = [0.0, 0.25, 0.5, 0.75][level]
    atmospheric = np.full_like(image, 0.92)
    return np.clip((1 - alpha) * image + alpha * atmospheric, 0, 1)


def _rain(image, level, rng):
    count = [0, 80, 180, 320][level]
    output = image.copy()
    h, w = image.shape[:2]
    for _ in range(count):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(8, 24))
        cv2.line(output, (x, y), (min(w - 1, x + 3), min(h - 1, y + length)),
                 (1.0, 1.0, 1.0), 1, cv2.LINE_AA)
    return np.clip(0.82 * image + 0.18 * output, 0, 1)


def _snow(image, level, rng):
    count = [0, 120, 300, 600][level]
    output = image.copy()
    h, w = image.shape[:2]
    for _ in range(count):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        radius = int(rng.integers(1, 3))
        cv2.circle(output, (x, y), radius, (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
    return np.clip(0.88 * image + 0.12 * output, 0, 1)


def _proxies(clean, individual):
    clean_gray = cv2.cvtColor(clean.astype(np.float32), cv2.COLOR_RGB2GRAY)
    image_gray = cv2.cvtColor(individual.astype(np.float32), cv2.COLOR_RGB2GRAY)
    clean_std = max(float(clean_gray.std()), 1e-6)
    positive_change = np.maximum(individual - clean, 0)
    return {
        "mean_luminance_change": float(image_gray.mean() - clean_gray.mean()),
        "contrast_loss": float(1.0 - image_gray.std() / clean_std),
        "mean_absolute_change": float(np.abs(individual - clean).mean()),
        "positive_change_fraction": float((positive_change > 0.03).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--samples-per-image", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    files = sorted(p for p in Path(args.clean_root).rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError("No clean images found")
    split_at = max(1, int(len(files) * (1 - args.val_fraction)))
    files = files[:split_at] if args.split == "train" else files[split_at:]
    if not files:
        raise ValueError(f"No images assigned to {args.split} split")
    output_root = Path(args.output_root) / args.split
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for source in files:
        clean = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if clean is None:
            continue
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for sample_index in range(args.samples_per_image):
            bins = [int(rng.integers(0, 4)) for _ in range(4)]
            low_image = _darken(clean, bins[0])
            haze_image = _haze(clean, bins[1])
            rain_image = _rain(clean, bins[2], rng)
            snow_image = _snow(clean, bins[3], rng)
            image = _snow(_rain(_haze(low_image, bins[1]), bins[2], rng), bins[3], rng)
            name = f"{source.stem}_{sample_index:04d}.png"
            output_path = output_root / name
            bgr = cv2.cvtColor(np.clip(image * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), bgr)
            records.append({
                "image_path": str(output_path),
                "source_image": str(source),
                "severity_bins": bins,
                "generation_parameters": {
                    "low_gamma": [1.0, 1.35, 1.9, 2.6][bins[0]],
                    "haze_alpha": [0.0, 0.25, 0.5, 0.75][bins[1]],
                    "rain_count": [0, 80, 180, 320][bins[2]],
                    "snow_count": [0, 120, 300, 600][bins[3]],
                },
                "severity_proxies": {
                    "low": _proxies(clean, low_image),
                    "haze": _proxies(clean, haze_image),
                    "rain": _proxies(clean, rain_image),
                    "snow": _proxies(clean, snow_image),
                },
                "label_note": "Rendering bins are provisional until proxy monotonicity and human perceptual checks pass.",
            })
    with manifest_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"split": args.split, "records": len(records), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
