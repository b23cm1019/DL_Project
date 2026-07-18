#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basicsr.archs import build_network
from basicsr.utils.options import yaml_load


SETTING_TO_VECTOR = {
    "low": [1, 0, 0, 0],
    "haze": [0, 1, 0, 0],
    "rain": [0, 0, 1, 0],
    "snow": [0, 0, 0, 1],
    "low_haze": [1, 1, 0, 0],
    "low_rain": [1, 0, 1, 0],
    "low_snow": [1, 0, 0, 1],
    "haze_rain": [0, 1, 1, 0],
    "haze_snow": [0, 1, 0, 1],
    "low_haze_rain": [1, 1, 1, 0],
    "low_haze_snow": [1, 1, 0, 1],
}

PRIMITIVE_NAMES = ["low", "haze", "rain", "snow"]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Row C degradation-classifier predictions on CDD-11."
    )
    parser.add_argument(
        "--opt",
        default="options/cdd_experiments/pretrain_multilabel.yml",
        help="Option YAML used to build network_g and network_dc.",
    )
    parser.add_argument(
        "--encoder_path",
        required=True,
        help="Path to net_g checkpoint used during classifier pretraining.",
    )
    parser.add_argument(
        "--classifier_path",
        required=True,
        help="Path to net_dc checkpoint used during classifier pretraining.",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Path to CDD-11 split root containing low/haze/rain/... subfolders.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where CSV/JSON/plots will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for converting sigmoid probabilities to binary predictions.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional cap per setting. Use 0 to analyze all images.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use, e.g. cuda or cpu.",
    )
    return parser.parse_args()


def load_checkpoint_state(load_path: Path, preferred_key: str | None) -> dict[str, Any]:
    checkpoint = torch.load(load_path, map_location="cpu")
    if preferred_key is not None and isinstance(checkpoint, dict) and preferred_key in checkpoint:
        return checkpoint[preferred_key]
    if isinstance(checkpoint, dict):
        for fallback_key in ("params", "params_ema", "state_dict"):
            if fallback_key in checkpoint:
                return checkpoint[fallback_key]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format in {load_path}")


def register_decoder_hooks(net_g: torch.nn.Module, hook_name: str) -> tuple[list[Any], list[torch.utils.hooks.RemovableHandle]]:
    hook_outputs: list[Any] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def hook_forward_fn(module: torch.nn.Module, module_input: Any, module_output: Any) -> None:
        if isinstance(module_output, tuple):
            module_output = module_output[-1]
        hook_outputs.append(module_output)

    for name, module in net_g.named_modules():
        if hook_name in name and name.count(".") == 1:
            handles.append(module.register_forward_hook(hook_forward_fn))

    if not handles:
        raise RuntimeError(f"No hook modules found using hook name '{hook_name}'.")

    return hook_outputs, handles


def load_image_tensor(image_path: Path) -> torch.Tensor:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


def pad_to_window_size(image: torch.Tensor, network_g_opt: dict[str, Any]) -> torch.Tensor:
    if "window_size" not in network_g_opt:
        return image

    _, _, h, w = image.shape
    window_size = network_g_opt.get("window_size", max(h, w))
    enc_blk_nums = network_g_opt.get("enc_blk_nums", [])
    downsample_multiple = 2 ** len(enc_blk_nums)
    window_size = math.lcm(int(window_size), max(1, int(downsample_multiple)))

    pad_h = 0 if h % window_size == 0 else window_size - h % window_size
    pad_w = 0 if w % window_size == 0 else window_size - w % window_size
    if pad_h == 0 and pad_w == 0:
        return image
    return F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")


def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def save_per_image_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "setting",
        "image_id",
        "gt_low",
        "gt_haze",
        "gt_rain",
        "gt_snow",
        "prob_low",
        "prob_haze",
        "prob_rain",
        "prob_snow",
        "pred_low",
        "pred_haze",
        "pred_rain",
        "pred_snow",
        "exact_match",
        "gt_count",
        "pred_count",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_setting_summary_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "setting",
        "num_images",
        "exact_match_acc",
        "cardinality_acc",
        "mean_prob_low",
        "mean_prob_haze",
        "mean_prob_rain",
        "mean_prob_snow",
        "fp_rate_on_single",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_primitive_metrics_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = ["primitive", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "support"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_mean_probability_heatmap(setting_rows: list[dict[str, Any]], out_path: Path) -> None:
    settings = [row["setting"] for row in setting_rows]
    matrix = np.array(
        [
            [
                row["mean_prob_low"],
                row["mean_prob_haze"],
                row["mean_prob_rain"],
                row["mean_prob_snow"],
            ]
            for row in setting_rows
        ],
        dtype=np.float32,
    )

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(PRIMITIVE_NAMES)))
    ax.set_xticklabels(PRIMITIVE_NAMES)
    ax.set_yticks(range(len(settings)))
    ax.set_yticklabels(settings)
    ax.set_title("Mean Sigmoid Probability per Setting")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_exact_match_bar(setting_rows: list[dict[str, Any]], out_path: Path) -> None:
    settings = [row["setting"] for row in setting_rows]
    exact_vals = [row["exact_match_acc"] for row in setting_rows]
    card_vals = [row["cardinality_acc"] for row in setting_rows]

    x = np.arange(len(settings))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, exact_vals, width=width, label="Exact Match")
    ax.bar(x + width / 2, card_vals, width=width, label="Cardinality Match")
    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=40, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Classifier Accuracy by Setting")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    opt_path = ROOT / args.opt if not Path(args.opt).is_absolute() else Path(args.opt)
    encoder_path = Path(args.encoder_path)
    classifier_path = Path(args.classifier_path)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not opt_path.exists():
        raise FileNotFoundError(f"Option file not found: {opt_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {classifier_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    opt = yaml_load(str(opt_path))
    network_g_opt = dict(opt["network_g"])
    network_dc_opt = dict(opt["network_dc"])
    hook_name = opt.get("hook_names", "decoder")

    device = torch.device(args.device)

    net_g = build_network(network_g_opt).to(device)
    net_dc = build_network(network_dc_opt).to(device)

    encoder_state = load_checkpoint_state(encoder_path, preferred_key="params")
    classifier_state = load_checkpoint_state(classifier_path, preferred_key="params")
    net_g.load_state_dict(encoder_state, strict=False)
    net_dc.load_state_dict(classifier_state, strict=True)

    net_g.eval()
    net_dc.eval()

    hook_outputs, handles = register_decoder_hooks(net_g, hook_name)

    settings = [name for name in SETTING_TO_VECTOR if (data_root / name).is_dir()]
    if not settings:
        raise RuntimeError(f"No expected CDD-11 setting folders were found under {data_root}")

    per_image_rows: list[dict[str, Any]] = []
    setting_summary_rows: list[dict[str, Any]] = []

    primitive_counts = {
        primitive: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for primitive in PRIMITIVE_NAMES
    }

    overall_exact = 0
    overall_card = 0
    overall_images = 0

    with torch.no_grad():
        for setting in settings:
            gt_vector = np.array(SETTING_TO_VECTOR[setting], dtype=np.int64)
            setting_dir = data_root / setting
            image_paths = sorted(
                [path for path in setting_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
            )
            if args.max_images > 0:
                image_paths = image_paths[: args.max_images]
            if not image_paths:
                continue

            probs_for_setting: list[np.ndarray] = []
            exact_matches = 0
            cardinality_matches = 0
            extra_fp_on_single = 0

            for image_path in image_paths:
                lq = load_image_tensor(image_path)
                lq = pad_to_window_size(lq, network_g_opt).to(device)

                hook_outputs.clear()
                _ = net_g(lq, hook=True)
                logits = net_dc(lq, hook_outputs[::-1])
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                preds = (probs >= args.threshold).astype(np.int64)

                exact_match = int(np.array_equal(preds, gt_vector))
                gt_count = int(gt_vector.sum())
                pred_count = int(preds.sum())
                card_match = int(gt_count == pred_count)

                overall_exact += exact_match
                overall_card += card_match
                overall_images += 1

                exact_matches += exact_match
                cardinality_matches += card_match
                probs_for_setting.append(probs)

                if gt_count == 1:
                    extra_fp_on_single += int(np.maximum(preds - gt_vector, 0).sum() > 0)

                for idx, primitive in enumerate(PRIMITIVE_NAMES):
                    gt_val = int(gt_vector[idx])
                    pred_val = int(preds[idx])
                    if gt_val == 1 and pred_val == 1:
                        primitive_counts[primitive]["tp"] += 1
                    elif gt_val == 0 and pred_val == 1:
                        primitive_counts[primitive]["fp"] += 1
                    elif gt_val == 0 and pred_val == 0:
                        primitive_counts[primitive]["tn"] += 1
                    else:
                        primitive_counts[primitive]["fn"] += 1

                per_image_rows.append(
                    {
                        "setting": setting,
                        "image_id": image_path.stem,
                        "gt_low": int(gt_vector[0]),
                        "gt_haze": int(gt_vector[1]),
                        "gt_rain": int(gt_vector[2]),
                        "gt_snow": int(gt_vector[3]),
                        "prob_low": float(probs[0]),
                        "prob_haze": float(probs[1]),
                        "prob_rain": float(probs[2]),
                        "prob_snow": float(probs[3]),
                        "pred_low": int(preds[0]),
                        "pred_haze": int(preds[1]),
                        "pred_rain": int(preds[2]),
                        "pred_snow": int(preds[3]),
                        "exact_match": exact_match,
                        "gt_count": gt_count,
                        "pred_count": pred_count,
                    }
                )

            mean_probs = np.mean(np.stack(probs_for_setting, axis=0), axis=0)
            is_single_setting = int(gt_vector.sum() == 1)
            fp_rate_on_single = safe_div(extra_fp_on_single, len(image_paths)) if is_single_setting else 0.0

            setting_summary_rows.append(
                {
                    "setting": setting,
                    "num_images": len(image_paths),
                    "exact_match_acc": safe_div(exact_matches, len(image_paths)),
                    "cardinality_acc": safe_div(cardinality_matches, len(image_paths)),
                    "mean_prob_low": float(mean_probs[0]),
                    "mean_prob_haze": float(mean_probs[1]),
                    "mean_prob_rain": float(mean_probs[2]),
                    "mean_prob_snow": float(mean_probs[3]),
                    "fp_rate_on_single": float(fp_rate_on_single),
                }
            )

    for handle in handles:
        handle.remove()

    primitive_metric_rows: list[dict[str, Any]] = []
    macro_f1 = 0.0
    macro_precision = 0.0
    macro_recall = 0.0
    for primitive in PRIMITIVE_NAMES:
        counts = primitive_counts[primitive]
        precision = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        f1 = safe_div(2 * precision * recall, precision + recall)
        support = counts["tp"] + counts["fn"]
        primitive_metric_rows.append(
            {
                "primitive": primitive,
                "tp": counts["tp"],
                "fp": counts["fp"],
                "tn": counts["tn"],
                "fn": counts["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    macro_precision /= len(PRIMITIVE_NAMES)
    macro_recall /= len(PRIMITIVE_NAMES)
    macro_f1 /= len(PRIMITIVE_NAMES)

    total_tp = sum(primitive_counts[p]["tp"] for p in PRIMITIVE_NAMES)
    total_fp = sum(primitive_counts[p]["fp"] for p in PRIMITIVE_NAMES)
    total_fn = sum(primitive_counts[p]["fn"] for p in PRIMITIVE_NAMES)
    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall = safe_div(total_tp, total_tp + total_fn)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    summary = {
        "opt": str(opt_path),
        "encoder_path": str(encoder_path),
        "classifier_path": str(classifier_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "threshold": args.threshold,
        "num_settings": len(setting_summary_rows),
        "num_images": overall_images,
        "overall_exact_match_acc": safe_div(overall_exact, overall_images),
        "overall_cardinality_acc": safe_div(overall_card, overall_images),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "settings": setting_summary_rows,
        "primitive_metrics": primitive_metric_rows,
    }

    save_per_image_csv(per_image_rows, output_dir / "per_image_predictions.csv")
    save_setting_summary_csv(setting_summary_rows, output_dir / "per_setting_summary.csv")
    save_primitive_metrics_csv(primitive_metric_rows, output_dir / "primitive_metrics.csv")
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    plot_mean_probability_heatmap(setting_summary_rows, output_dir / "mean_probability_heatmap.png")
    plot_exact_match_bar(setting_summary_rows, output_dir / "setting_accuracy.png")

    print("Classifier diagnosis complete.")
    print(f"Images analyzed: {overall_images}")
    print(f"Overall exact-match accuracy: {summary['overall_exact_match_acc']:.4f}")
    print(f"Overall cardinality accuracy: {summary['overall_cardinality_acc']:.4f}")
    print(f"Micro F1: {summary['micro_f1']:.4f}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
