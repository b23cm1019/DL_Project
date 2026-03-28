#!/usr/bin/env python3
"""Select qualitative examples with the largest gains over Row B.

This script computes per-image PSNR/SSIM from saved test outputs, ranks images
by gain over Row B, and exports the top examples into report-friendly folders.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim


@dataclass
class ModelSpec:
    name: str
    vis_root: Path
    suffix: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of images to export per degradation type for each challenger.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/data/datasets/CDD-11_test/clear"),
    )
    parser.add_argument(
        "--lq-root",
        type=Path,
        default=Path("/data/datasets/CDD-11_test"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/qualitative_subsets"),
    )
    return parser.parse_args()


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img[np.newaxis, ...]


def calculate_psnr_pair(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(
        calculate_psnr(
            pred,
            gt,
            crop_border=0,
            input_order="BHWC",
            test_y_channel=False,
            image_range=255,
        )
    )


def calculate_ssim_pair(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(
        calculate_ssim(
            pred,
            gt,
            crop_border=0,
            input_order="BHWC",
            test_y_channel=False,
            image_range=255,
        )
    )


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def export_case(
    output_root: Path,
    challenger: str,
    degradation: str,
    img_id: str,
    lq_root: Path,
    gt_root: Path,
    models: dict[str, ModelSpec],
) -> None:
    lq_src = lq_root / degradation / f"{img_id}.png"
    gt_src = gt_root / f"{img_id}.png"
    copy_if_exists(lq_src, output_root / challenger / "input" / degradation / f"{img_id}.png")
    copy_if_exists(gt_src, output_root / challenger / "gt" / degradation / f"{img_id}.png")
    for model_name, spec in models.items():
        pred_src = spec.vis_root / degradation / f"{img_id}_{spec.suffix}.png"
        copy_if_exists(
            pred_src,
            output_root / challenger / model_name / degradation / f"{img_id}_{spec.suffix}.png",
        )


def main() -> None:
    args = parse_args()

    models = {
        "row_b": ModelSpec(
            name="row_b",
            vis_root=Path("results/Test_Row_B/visualization"),
            suffix="row_b",
        ),
        "row_c": ModelSpec(
            name="row_c",
            vis_root=Path("results/Test_Row_C/visualization"),
            suffix="row_c",
        ),
        "row_d": ModelSpec(
            name="row_d",
            vis_root=Path("results/Test_Row_D/visualization"),
            suffix="row_d",
        ),
    }

    baseline = models["row_b"]
    degradations = sorted(
        d.name for d in baseline.vis_root.iterdir() if d.is_dir()
    )

    summary_root = args.output_root
    summary_root.mkdir(parents=True, exist_ok=True)
    manifest_path = summary_root / "selection_manifest.csv"

    rows: list[dict[str, object]] = []
    image_cache: dict[Path, np.ndarray] = {}
    psnr_cache: dict[tuple[str, str, str], float] = {}
    ssim_cache: dict[tuple[str, str, str], float] = {}

    def get_image(path: Path) -> np.ndarray:
        if path not in image_cache:
            image_cache[path] = read_rgb(path)
        return image_cache[path]

    def get_psnr(model_name: str, degradation: str, img_id: str) -> float:
        key = (model_name, degradation, img_id)
        if key in psnr_cache:
            return psnr_cache[key]

        spec = models[model_name]
        pred_path = spec.vis_root / degradation / f"{img_id}_{spec.suffix}.png"
        gt_path = args.gt_root / f"{img_id}.png"
        psnr_cache[key] = calculate_psnr_pair(get_image(pred_path), get_image(gt_path))
        return psnr_cache[key]

    def get_ssim(model_name: str, degradation: str, img_id: str) -> float:
        key = (model_name, degradation, img_id)
        if key in ssim_cache:
            return ssim_cache[key]

        spec = models[model_name]
        pred_path = spec.vis_root / degradation / f"{img_id}_{spec.suffix}.png"
        gt_path = args.gt_root / f"{img_id}.png"
        ssim_cache[key] = calculate_ssim_pair(get_image(pred_path), get_image(gt_path))
        return ssim_cache[key]

    for degradation in degradations:
        baseline_dir = baseline.vis_root / degradation
        base_files = sorted(baseline_dir.glob(f"*_{baseline.suffix}.png"))
        base_ids = [p.name[: -len(f"_{baseline.suffix}.png")] for p in base_files]

        base_metrics: dict[str, tuple[float, float]] = {}
        for img_id in base_ids:
            base_metrics[img_id] = (
                get_psnr("row_b", degradation, img_id),
                0.0,
            )

        for challenger_name in ("row_c", "row_d"):
            spec = models[challenger_name]
            challenger_dir = spec.vis_root / degradation
            if not challenger_dir.exists():
                continue

            ranked: list[dict[str, object]] = []
            for img_id in base_ids:
                pred_path = challenger_dir / f"{img_id}_{spec.suffix}.png"
                gt_path = args.gt_root / f"{img_id}.png"
                if not pred_path.exists() or not gt_path.exists():
                    continue

                base_psnr, base_ssim = base_metrics[img_id]
                cur_psnr = get_psnr(challenger_name, degradation, img_id)
                ranked.append(
                    {
                        "challenger": challenger_name,
                        "degradation": degradation,
                        "image_id": img_id,
                        "row_b_psnr": base_psnr,
                        "row_b_ssim": "",
                        f"{challenger_name}_psnr": cur_psnr,
                        f"{challenger_name}_ssim": "",
                        "psnr_gain": cur_psnr - base_psnr,
                        "ssim_gain": "",
                    }
                )

            ranked.sort(key=lambda x: float(x["psnr_gain"]), reverse=True)

            selected = ranked[: args.top_k]
            for rank, item in enumerate(selected, start=1):
                item["row_b_ssim"] = get_ssim("row_b", degradation, str(item["image_id"]))
                item[f"{challenger_name}_ssim"] = get_ssim(
                    challenger_name, degradation, str(item["image_id"])
                )
                item["ssim_gain"] = (
                    float(item[f"{challenger_name}_ssim"]) - float(item["row_b_ssim"])
                )
                item["rank"] = rank
                rows.append(item)
                export_case(
                    output_root=summary_root,
                    challenger=challenger_name,
                    degradation=degradation,
                    img_id=str(item["image_id"]),
                    lq_root=args.lq_root,
                    gt_root=args.gt_root,
                    models=models,
                )

    fieldnames = [
        "challenger",
        "degradation",
        "rank",
        "image_id",
        "row_b_psnr",
        "row_b_ssim",
        "row_c_psnr",
        "row_c_ssim",
        "row_d_psnr",
        "row_d_ssim",
        "psnr_gain",
        "ssim_gain",
    ]

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    counts = {}
    for row in rows:
        counts.setdefault(row["challenger"], 0)
        counts[row["challenger"]] += 1

    print(f"Wrote manifest to {manifest_path}")
    for challenger, count in sorted(counts.items()):
        print(f"{challenger}: exported {count} selections")


if __name__ == "__main__":
    main()
