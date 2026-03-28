#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
CARD_DIR = REPORT_DIR / "qualitative_cards"
for path in (REPORT_DIR, FIG_DIR, TABLE_DIR, CARD_DIR):
    path.mkdir(parents=True, exist_ok=True)

DEGRADATIONS = [
    "low",
    "haze",
    "rain",
    "snow",
    "low_haze",
    "low_rain",
    "low_snow",
    "haze_rain",
    "haze_snow",
    "low_haze_rain",
    "low_haze_snow",
]

TEST_LOGS = {
    "row_a": ROOT / "results/Test_Row_B_archived_20260327_101719/test_Test_Row_B_20260326_030742.log",
    "row_b": ROOT / "results/Test_Row_B/test_Test_Row_B_20260327_232612.log",
    "row_c": ROOT / "results/Test_Row_C/test_Test_Row_C_20260327_232645.log",
    "row_d": ROOT / "results/Test_Row_D_archived_20260327_232802/test_Test_Row_D_20260327_152041.log",
}

PRETRAIN_LOGS = {
    "row_b": [
        ROOT / "experiments/Row_B_Baseline_Pretrain_CDD11/train_Row_B_Baseline_Pretrain_CDD11_20260326_181751.log",
        ROOT / "experiments/Row_B_Baseline_Pretrain_CDD11/train_Row_B_Baseline_Pretrain_CDD11_20260326_192904.log",
        ROOT / "experiments/Row_B_Baseline_Pretrain_CDD11/train_Row_B_Baseline_Pretrain_CDD11_20260326_195026.log",
    ],
    "row_c": [
        ROOT / "experiments/Row_C_MultiLabel_Pretrain_CDD11/train_Row_C_MultiLabel_Pretrain_CDD11_20260326_183609.log",
        ROOT / "experiments/Row_C_MultiLabel_Pretrain_CDD11/train_Row_C_MultiLabel_Pretrain_CDD11_20260326_195100.log",
    ],
}

FINETUNE_LOGS = {
    "row_b": [ROOT / "experiments/Row_B_Baseline_Finetune_CDD11/train_Row_B_Baseline_Finetune_CDD11_20260327_004559.log"],
    "row_c": [ROOT / "experiments/Row_C_MultiLabel_Finetune_CDD11_archived_20260327_004259/train_Row_C_MultiLabel_Finetune_CDD11_20260327_004015.log"],
    "row_d": [
        ROOT / "experiments/Row_D_Prompt_Finetune_CDD11/train_Row_D_Prompt_Finetune_CDD11_20260327_005239.log",
        ROOT / "experiments/Row_D_Prompt_Finetune_CDD11/train_Row_D_Prompt_Finetune_CDD11_20260327_084541.log",
    ],
}

VIS_ROOTS = {
    "row_b": ROOT / "results/Test_Row_B/visualization",
    "row_c": ROOT / "results/Test_Row_C/visualization",
    "row_d": ROOT / "results/Test_Row_D/visualization",
}

SUFFIX = {"row_b": "row_b", "row_c": "row_c", "row_d": "row_d"}
GT_ROOT = Path("/data/datasets/CDD-11_test/clear")
LQ_ROOT = Path("/data/datasets/CDD-11_test")
MODEL_ORDER = ["row_b", "row_c", "row_d"]
MODEL_LABELS = {"row_a": "Row A", "row_b": "Row B", "row_c": "Row C", "row_d": "Row D"}
MODEL_COLORS = {"row_a": "#9aa0a6", "row_b": "#355070", "row_c": "#2a9d8f", "row_d": "#e76f51"}
MODEL_MARKERS = {"row_b": "o", "row_c": "s", "row_d": "D"}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
    }
)


def parse_test_metrics(log_path: Path) -> dict[str, dict[str, float]]:
    text = log_path.read_text()
    entries: dict[str, dict[str, float]] = {}
    current = None
    for line in text.splitlines():
        m = re.search(r"Testing ([a-z_]+)\.\.\.", line)
        if m:
            current = m.group(1)
            entries[current] = {}
            continue
        if current is None:
            continue
        m = re.search(r"# psnr:\s*([0-9.]+)", line)
        if m:
            entries[current]["psnr"] = float(m.group(1))
        m = re.search(r"# ssim:\s*([0-9.]+)", line)
        if m:
            entries[current]["ssim"] = float(m.group(1))
    return entries


def average_metric(metrics: dict[str, dict[str, float]], key: str) -> float:
    return sum(metrics[d][key] for d in DEGRADATIONS) / len(DEGRADATIONS)


def friendly_name(degradation: str) -> str:
    return degradation.replace("_", " + ")


def save_csv(rows: list[dict[str, object]], fieldnames: list[str], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def moving_average(xs: list[float], window: int) -> np.ndarray:
    if len(xs) < window:
        return np.array(xs)
    kernel = np.ones(window) / window
    return np.convolve(np.array(xs), kernel, mode="valid")


def ema_smooth(xs: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    if len(xs) == 0:
        return xs
    out = np.empty_like(xs, dtype=float)
    out[0] = xs[0]
    for i in range(1, len(xs)):
        out[i] = alpha * xs[i] + (1 - alpha) * out[i - 1]
    return out


def series_to_arrays(points: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([x for x, _ in points], dtype=float), np.array([y for _, y in points], dtype=float)


def parse_loss_series(log_paths: list[Path]) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, dict[int, float]] = defaultdict(dict)
    pattern = re.compile(r"iter:\s*([0-9,]+).*?l_pix:\s*([0-9.eE+-]+)(?:\s+l_classify:\s*([0-9.eE+-]+))?")
    for path in log_paths:
        for line in path.read_text(errors="ignore").splitlines():
            m = pattern.search(line)
            if not m:
                continue
            itr = int(m.group(1).replace(",", ""))
            series["l_pix"][itr] = float(m.group(2))
            if m.group(3) is not None:
                series["l_classify"][itr] = float(m.group(3))
    out: dict[str, list[tuple[int, float]]] = {}
    for key, values in series.items():
        out[key] = sorted(values.items())
    return out


def plot_grouped_metrics(all_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    order = sorted(
        DEGRADATIONS,
        key=lambda d: all_metrics["row_c"][d]["psnr"] - all_metrics["row_b"][d]["psnr"],
        reverse=True,
    )
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.4), constrained_layout=True)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=MODEL_MARKERS[row],
            linestyle="",
            markersize=8,
            markerfacecolor=MODEL_COLORS[row],
            markeredgecolor="white",
            markeredgewidth=0.9,
            label=MODEL_LABELS[row],
        )
        for row in MODEL_ORDER
    ]

    for ax, metric, title in [
        (axes[0], "psnr", "Per-setting PSNR"),
        (axes[1], "ssim", "Per-setting SSIM"),
    ]:
        for i, degradation in enumerate(order):
            vals = [all_metrics[row][degradation][metric] for row in MODEL_ORDER]
            if "_" in degradation:
                ax.axhspan(i - 0.46, i + 0.46, color="#f7f3ed", zorder=0)
            ax.hlines(i, min(vals), max(vals), color="#d0d4da", linewidth=2.4, zorder=1)
            for row in MODEL_ORDER:
                ax.scatter(
                    all_metrics[row][degradation][metric],
                    i,
                    s=78,
                    marker=MODEL_MARKERS[row],
                    color=MODEL_COLORS[row],
                    edgecolor="white",
                    linewidth=0.9,
                    zorder=3,
                )
            if metric == "psnr":
                gain = all_metrics["row_c"][degradation]["psnr"] - all_metrics["row_b"][degradation]["psnr"]
                ax.text(
                    max(vals) + 0.06,
                    i,
                    f"{gain:+.2f}",
                    va="center",
                    fontsize=8.3,
                    color="#2a9d8f",
                    fontweight="bold",
                )

        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels([friendly_name(d) for d in order])
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("dB" if metric == "psnr" else "SSIM")

    axes[0].legend(handles=legend_handles, loc="lower right", fontsize=9, ncol=3)
    axes[0].text(
        0.0,
        1.02,
        "Compound degradations are lightly shaded; right-side labels show Row C - Row B gain (dB).",
        transform=axes[0].transAxes,
        fontsize=8.4,
        color="#666666",
    )
    fig.savefig(FIG_DIR / "grouped_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gain_bars(all_metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    gain_bc = {d: all_metrics["row_c"][d]["psnr"] - all_metrics["row_b"][d]["psnr"] for d in DEGRADATIONS}
    gain_dc = {d: all_metrics["row_d"][d]["psnr"] - all_metrics["row_c"][d]["psnr"] for d in DEGRADATIONS}
    order_bc = sorted(DEGRADATIONS, key=lambda d: gain_bc[d], reverse=True)
    order_dc = sorted(DEGRADATIONS, key=lambda d: gain_dc[d], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8), constrained_layout=True)
    for ax, order, gains, title, avg_gain in [
        (
            axes[0],
            order_bc,
            gain_bc,
            "Row C - Row B (multi-label BCE gain)",
            average_metric(all_metrics["row_c"], "psnr") - average_metric(all_metrics["row_b"], "psnr"),
        ),
        (
            axes[1],
            order_dc,
            gain_dc,
            "Row D - Row C (prompt gain)",
            average_metric(all_metrics["row_d"], "psnr") - average_metric(all_metrics["row_c"], "psnr"),
        ),
    ]:
        vals = [gains[d] for d in order]
        colors = []
        for degradation, val in zip(order, vals):
            if title.startswith("Row C"):
                colors.append("#2a9d8f" if "_" not in degradation else "#7dcfb6")
            else:
                if val >= 0:
                    colors.append("#e76f51" if "_" not in degradation else "#f4a261")
                else:
                    colors.append("#9c6644" if "_" not in degradation else "#c97b63")
        y = np.arange(len(order))
        ax.barh(y, vals, color=colors, edgecolor="none", height=0.7)
        ax.axvline(0.0, color="#4d4d4d", linewidth=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels([friendly_name(d) for d in order])
        ax.invert_yaxis()
        ax.set_xlabel("PSNR gain (dB)")
        ax.set_title(f"{title}\nAverage gain: {avg_gain:+.4f} dB", fontsize=11)
        ax.grid(axis="x")
        for yi, val in zip(y, vals):
            ha = "left" if val >= 0 else "right"
            offset = 0.015 if val >= 0 else -0.015
            ax.text(val + offset, yi, f"{val:+.3f}", va="center", ha=ha, fontsize=8.2)

    best_bc = order_bc[0]
    best_dc = order_dc[0]
    worst_dc = order_dc[-1]
    axes[0].text(
        0.99,
        0.03,
        f"Best BCE gain: {friendly_name(best_bc)} ({gain_bc[best_bc]:+.3f} dB)",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="#1f6f63",
    )
    axes[1].text(
        0.99,
        0.03,
        f"Best prompt gain: {friendly_name(best_dc)} ({gain_dc[best_dc]:+.3f} dB)\nLargest drop: {friendly_name(worst_dc)} ({gain_dc[worst_dc]:+.3f} dB)",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="#7f5539",
    )
    fig.savefig(FIG_DIR / "psnr_gains.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves() -> None:
    pretrain_data = {row: parse_loss_series(PRETRAIN_LOGS[row]) for row in ("row_b", "row_c")}
    finetune_data = {row: parse_loss_series(FINETUNE_LOGS[row]) for row in ("row_b", "row_c", "row_d")}

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    pretrain_specs = [
        (axes[0], "l_pix", "Pretraining Pixel Loss", "Pixel Loss", 0.08),
        (axes[1], "l_classify", "Pretraining Classification Loss", "Classification Loss", 0.08),
    ]
    for ax, key, title, ylabel, alpha in pretrain_specs:
        for row in ("row_b", "row_c"):
            iters, vals = series_to_arrays(pretrain_data[row][key])
            ax.plot(iters, vals, color=MODEL_COLORS[row], alpha=0.10, linewidth=0.8)
            ax.plot(iters, ema_smooth(vals, alpha=alpha), color=MODEL_COLORS[row], linewidth=2.8, label=MODEL_LABELS[row])
        ax.axvline(5000, color="#8d99ae", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Pretraining Iteration")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(alpha=0.22)
        ax.legend(loc="upper right")
    axes[0].text(
        0.02,
        0.05,
        "Thin traces are the actual logged losses.\nBold traces are EMA-smoothed for readability.",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#666666",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fafafa", edgecolor="#dddddd"),
    )
    axes[1].text(
        0.98,
        0.05,
        "Row B and Row C use different classification objectives.\nUse this panel to compare stability, not absolute scale.",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#666666",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fafafa", edgecolor="#dddddd"),
    )
    fig.savefig(FIG_DIR / "pretrain_losses.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.8, 5.6), constrained_layout=True)
    for row in ("row_b", "row_c", "row_d"):
        iters, vals = series_to_arrays(finetune_data[row]["l_pix"])
        ax.plot(iters, vals, color=MODEL_COLORS[row], alpha=0.09, linewidth=0.8)
        ax.plot(iters, ema_smooth(vals, alpha=0.05), color=MODEL_COLORS[row], linewidth=2.8, label=MODEL_LABELS[row])
    for step in (25000, 50000, 75000, 100000):
        ax.axvline(step, color="#d9dde3", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_title("Finetuning Pixel Loss")
    ax.set_xlabel("Finetuning Iteration")
    ax.set_ylabel("Pixel Loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.05,
        "This uses the actual logged l_pix values from the completed runs.\nThe smoothed overlay is only for easier trend reading.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.7,
        color="#666666",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fafafa", edgecolor="#dddddd"),
    )
    fig.savefig(FIG_DIR / "finetune_losses.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.8, 5.8), constrained_layout=True)
    ax.axvspan(0, 25000, color="#f6efe7", alpha=0.95, zorder=0)
    ax.axvspan(25000, 125000, color="#eef5fb", alpha=0.95, zorder=0)
    ax.axvline(25000, color="#8d99ae", linestyle="--", linewidth=1.3)
    ax.text(12500, 0.97, "Pretraining", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11, fontweight="bold", color="#6b4f3b")
    ax.text(75000, 0.97, "Finetuning", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11, fontweight="bold", color="#36506c")

    for row in ("row_b", "row_c"):
        pt_iters, pt_vals = series_to_arrays(pretrain_data[row]["l_pix"])
        ft_iters, ft_vals = series_to_arrays(finetune_data[row]["l_pix"])
        ax.plot(pt_iters, ema_smooth(pt_vals, alpha=0.08), color=MODEL_COLORS[row], linewidth=2.3, linestyle="--", label=f"{MODEL_LABELS[row]} pretrain")
        ax.plot(ft_iters + 25000, ema_smooth(ft_vals, alpha=0.05), color=MODEL_COLORS[row], linewidth=2.8, linestyle="-", label=f"{MODEL_LABELS[row]} finetune")

    d_iters, d_vals = series_to_arrays(finetune_data["row_d"]["l_pix"])
    ax.plot(d_iters + 25000, ema_smooth(d_vals, alpha=0.05), color=MODEL_COLORS["row_d"], linewidth=2.8, linestyle="-", label="Row D finetune")
    ax.text(
        0.98,
        0.08,
        "Row D has no separate pretraining stage in this setup.\nIt starts from the Row C pretrained backbone.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.7,
        color="#666666",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fafafa", edgecolor="#dddddd"),
    )
    ax.set_title("End-to-End Training Workflow (Pixel Loss)")
    ax.set_xlabel("Workflow Iteration  (0-25k pretrain, 25k-125k finetune)")
    ax.set_ylabel("Pixel Loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    fig.savefig(FIG_DIR / "workflow_training_curves.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def psnr_from_paths(pred_path: Path, gt_path: Path) -> float:
    pred = cv2.cvtColor(cv2.imread(str(pred_path)), cv2.COLOR_BGR2RGB).astype(np.float64)
    gt = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB).astype(np.float64)
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVu Sans:bold" if bold else "DejaVu Sans"
    return ImageFont.truetype(font_manager.findfont(family), size)


def fit_panel(img: Image.Image, target_w: int, target_h: int) -> tuple[Image.Image, float, tuple[int, int]]:
    src = img.copy()
    src.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (target_w, target_h), "#f3f5f7")
    offset = ((target_w - src.width) // 2, (target_h - src.height) // 2)
    panel.paste(src, offset)
    scale = min(target_w / img.width, target_h / img.height)
    return panel, scale, offset


def best_crop_box(degradation: str, img_id: str, baseline: str, challenger: str) -> tuple[int, int, int, int]:
    base = cv2.cvtColor(
        cv2.imread(str(VIS_ROOTS[baseline] / degradation / f"{img_id}_{SUFFIX[baseline]}.png")),
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32)
    challenger_img = cv2.cvtColor(
        cv2.imread(str(VIS_ROOTS[challenger] / degradation / f"{img_id}_{SUFFIX[challenger]}.png")),
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32)
    gt = cv2.cvtColor(cv2.imread(str(GT_ROOT / f"{img_id}.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
    improvement = np.mean(np.abs(base - gt) - np.abs(challenger_img - gt), axis=2)
    h, w = improvement.shape
    win = max(110, min(h, w) // 5)
    score = cv2.boxFilter(improvement, ddepth=-1, ksize=(win, win), normalize=False)
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    x0 = int(np.clip(cx - win // 2, 0, w - win))
    y0 = int(np.clip(cy - win // 2, 0, h - win))
    return x0, y0, win, win


def crop_to_panel(img: Image.Image, box: tuple[int, int, int, int], target_size: int) -> Image.Image:
    x0, y0, bw, bh = box
    crop = img.crop((x0, y0, x0 + bw, y0 + bh))
    crop = crop.resize((target_size, target_size), Image.Resampling.LANCZOS)
    return ImageOps.expand(crop, border=2, fill="#ffffff")


def best_case_for_degradation(degradation: str, challenger: str, baseline: str) -> tuple[str, float]:
    baseline_root = VIS_ROOTS[baseline] / degradation
    challenger_root = VIS_ROOTS[challenger] / degradation
    ids = sorted(p.name[:-len(f"_{SUFFIX[baseline]}.png")] for p in baseline_root.glob(f"*_{SUFFIX[baseline]}.png"))
    best_id = ids[0]
    best_gain = -1e9
    for img_id in ids:
        base_path = baseline_root / f"{img_id}_{SUFFIX[baseline]}.png"
        ch_path = challenger_root / f"{img_id}_{SUFFIX[challenger]}.png"
        if not ch_path.exists():
            continue
        if not base_path.exists():
            continue
        gain = image_psnr(degradation, img_id, challenger) - image_psnr(degradation, img_id, baseline)
        if gain > best_gain:
            best_gain = gain
            best_id = img_id
    return best_id, best_gain


def best_image_id(degradation: str, challenger: str, baseline: str) -> str:
    return best_case_for_degradation(degradation, challenger, baseline)[0]


@lru_cache(maxsize=None)
def image_psnr(degradation: str, img_id: str, model: str) -> float:
    pred_path = VIS_ROOTS[model] / degradation / f"{img_id}_{SUFFIX[model]}.png"
    return psnr_from_paths(pred_path, GT_ROOT / f"{img_id}.png")


def draw_centered_multiline(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    spacing: int = 4,
) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=spacing)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = x0 + (x1 - x0 - text_w) / 2
    y = y0 + (y1 - y0 - text_h) / 2
    draw.multiline_text((x, y), text, fill=fill, font=font, align="center", spacing=spacing)


def build_example_card(degradation: str, img_id: str, baseline: str, challenger: str, standalone: bool = False) -> Image.Image:
    labels = ["Input", "Row B", "Row C", "Row D", "GT"]
    images = {
        "Input": read_rgb(LQ_ROOT / degradation / f"{img_id}.png"),
        "Row B": read_rgb(VIS_ROOTS["row_b"] / degradation / f"{img_id}_{SUFFIX['row_b']}.png"),
        "Row C": read_rgb(VIS_ROOTS["row_c"] / degradation / f"{img_id}_{SUFFIX['row_c']}.png"),
        "Row D": read_rgb(VIS_ROOTS["row_d"] / degradation / f"{img_id}_{SUFFIX['row_d']}.png"),
        "GT": read_rgb(GT_ROOT / f"{img_id}.png"),
    }
    model_name_map = {"Row B": "row_b", "Row C": "row_c", "Row D": "row_d"}
    crop_box = best_crop_box(degradation, img_id, baseline, challenger)

    if standalone:
        full_w, full_h = 308, 220
        crop_size = 196
        gap = 26
        left = 34
        top = 88
        header_height = 58
        label_height = 56
        row_gap = 44
        footer = 34
        header_font = get_font(28, bold=True)
        body_font = get_font(22, bold=True)
        small_font = get_font(18)
        crop_border = 6
    else:
        full_w, full_h = 210, 150
        crop_size = 132
        gap = 18
        left = 22
        top = 64
        header_height = 42
        label_height = 34
        row_gap = 28
        footer = 36
        header_font = get_font(20, bold=True)
        body_font = get_font(13)
        small_font = get_font(12)
        crop_border = 4

    width = left * 2 + len(labels) * full_w + (len(labels) - 1) * gap
    height = header_height + label_height + full_h + row_gap + label_height + crop_size + footer
    card = Image.new("RGB", (width, height), "#fffdf8")
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=18, fill="#fffdf8", outline="#e6dccf", width=2)
    draw.rectangle((0, 0, width, header_height), fill="#243447")
    gain_value = image_psnr(degradation, img_id, challenger) - image_psnr(degradation, img_id, baseline)
    header = (
        f"{friendly_name(degradation)}  |  image {img_id}  |  "
        f"{MODEL_LABELS[challenger]} - {MODEL_LABELS[baseline]}: {gain_value:+.2f} dB"
    )
    draw.text((left, 12 if standalone else 10), header, fill="white", font=header_font)
    label_y0 = header_height
    full_row_y = label_y0 + label_height
    crop_label_y = full_row_y + full_h + row_gap
    crop_row_y = crop_label_y + label_height
    draw.text((left, label_y0 + 8), "Full image", fill="#5b5b5b", font=small_font)
    draw.text((left, crop_label_y + 8), "Zoom crop", fill="#5b5b5b", font=small_font)

    rect_color = "#ff7f11"
    for idx, label in enumerate(labels):
        x = left + idx * (full_w + gap)
        img = images[label]
        panel, scale, offset = fit_panel(img, full_w, full_h)
        if label != "GT":
            x0, y0, bw, bh = crop_box
            rx0 = int(x0 * scale + offset[0])
            ry0 = int(y0 * scale + offset[1])
            rx1 = int((x0 + bw) * scale + offset[0])
            ry1 = int((y0 + bh) * scale + offset[1])
            panel_draw = ImageDraw.Draw(panel)
            panel_draw.rectangle((rx0, ry0, rx1, ry1), outline=rect_color, width=crop_border)
        card.paste(panel, (x, full_row_y))

        title = label
        if label in model_name_map:
            title += f"\n{image_psnr(degradation, img_id, model_name_map[label]):.2f} dB"
        elif label == "GT":
            title += "\nreference"
        else:
            title += "\ndegraded"
        draw_centered_multiline(
            draw,
            (x, label_y0, x + full_w, label_y0 + label_height - 4),
            title,
            font=body_font,
            fill="#1b1b1b",
            spacing=4 if standalone else 2,
        )

        crop = crop_to_panel(img, crop_box, crop_size)
        crop_x = x + (full_w - crop.width) // 2
        crop_y = crop_row_y
        card.paste(crop, (crop_x, crop_y))

    return card


def make_qualitative_figure() -> list[tuple[str, str]]:
    selected = [
        ("haze", best_image_id("haze", "row_c", "row_b")),
        ("low_haze_rain", best_image_id("low_haze_rain", "row_d", "row_c")),
    ]
    cards = [
        build_example_card(selected[0][0], selected[0][1], baseline="row_b", challenger="row_c"),
        build_example_card(selected[1][0], selected[1][1], baseline="row_c", challenger="row_d"),
    ]
    width = max(card.size[0] for card in cards)
    gap = 26
    height = sum(card.size[1] for card in cards) + gap * (len(cards) - 1)
    canvas = Image.new("RGB", (width, height), "#f4efe8")
    y = 0
    for card in cards:
        canvas.paste(card, ((width - card.size[0]) // 2, y))
        y += card.size[1] + gap
    canvas.save(FIG_DIR / "qualitative_examples.png")
    return selected


def export_qualitative_cards(all_metrics: dict[str, dict[str, dict[str, float]]], top_k: int = 5) -> None:
    comparisons = [
        ("row_c", "row_b"),
        ("row_d", "row_c"),
        ("row_d", "row_b"),
    ]
    manifest_rows = []
    for challenger, baseline in comparisons:
        comparison_name = f"{challenger}_vs_{baseline}"
        out_dir = CARD_DIR / comparison_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for existing in out_dir.glob("*.png"):
            existing.unlink()
        degradation_order = sorted(
            DEGRADATIONS,
            key=lambda d: all_metrics[challenger][d]["psnr"] - all_metrics[baseline][d]["psnr"],
            reverse=True,
        )
        ranked = []
        for degradation in degradation_order[:top_k]:
            img_id, gain = best_case_for_degradation(degradation, challenger, baseline)
            ranked.append((degradation, img_id, gain))
        ranked.sort(key=lambda item: item[2], reverse=True)
        for rank, (degradation, img_id, gain) in enumerate(ranked[:top_k], start=1):
            card = build_example_card(degradation, img_id, baseline=baseline, challenger=challenger, standalone=True)
            filename = f"{rank:02d}_{degradation}_{img_id}_{challenger}_minus_{baseline}.png"
            card.save(out_dir / filename)
            manifest_rows.append(
                {
                    "comparison": comparison_name,
                    "rank": rank,
                    "degradation": degradation,
                    "image_id": img_id,
                    "gain_db": round(gain, 4),
                    "row_b_psnr": round(image_psnr(degradation, img_id, "row_b"), 4),
                    "row_c_psnr": round(image_psnr(degradation, img_id, "row_c"), 4),
                    "row_d_psnr": round(image_psnr(degradation, img_id, "row_d"), 4),
                    "file": filename,
                }
            )
    save_csv(
        manifest_rows,
        ["comparison", "rank", "degradation", "image_id", "gain_db", "row_b_psnr", "row_c_psnr", "row_d_psnr", "file"],
        CARD_DIR / "qualitative_card_manifest.csv",
    )


def write_table_snippets(all_metrics: dict[str, dict[str, dict[str, float]]], selected: list[tuple[str, str]]) -> None:
    rows = []
    for degradation in DEGRADATIONS:
        row = {"setting": degradation}
        for tag in ("row_a", "row_b", "row_c", "row_d"):
            row[f"{tag}_psnr"] = round(all_metrics[tag][degradation]["psnr"], 4)
            row[f"{tag}_ssim"] = round(all_metrics[tag][degradation]["ssim"], 4)
        row["c_minus_b_psnr"] = round(row["row_c_psnr"] - row["row_b_psnr"], 4)
        row["d_minus_c_psnr"] = round(row["row_d_psnr"] - row["row_c_psnr"], 4)
        rows.append(row)
    save_csv(
        rows,
        [
            "setting",
            "row_a_psnr",
            "row_a_ssim",
            "row_b_psnr",
            "row_b_ssim",
            "row_c_psnr",
            "row_c_ssim",
            "row_d_psnr",
            "row_d_ssim",
            "c_minus_b_psnr",
            "d_minus_c_psnr",
        ],
        TABLE_DIR / "metrics_table.csv",
    )

    with (TABLE_DIR / "selected_examples.txt").open("w") as f:
        for degradation, img_id in selected:
            f.write(f"{degradation}: {img_id}\n")


def main() -> None:
    all_metrics = {tag: parse_test_metrics(path) for tag, path in TEST_LOGS.items()}

    summary_rows = []
    for degradation in DEGRADATIONS:
        summary_rows.append(
            {
                "setting": degradation,
                "row_b_psnr": all_metrics["row_b"][degradation]["psnr"],
                "row_c_psnr": all_metrics["row_c"][degradation]["psnr"],
                "row_d_psnr": all_metrics["row_d"][degradation]["psnr"],
                "row_b_ssim": all_metrics["row_b"][degradation]["ssim"],
                "row_c_ssim": all_metrics["row_c"][degradation]["ssim"],
                "row_d_ssim": all_metrics["row_d"][degradation]["ssim"],
                "c_minus_b_psnr": all_metrics["row_c"][degradation]["psnr"] - all_metrics["row_b"][degradation]["psnr"],
                "d_minus_c_psnr": all_metrics["row_d"][degradation]["psnr"] - all_metrics["row_c"][degradation]["psnr"],
            }
        )

    summary_rows.append(
        {
            "setting": "average",
            "row_b_psnr": average_metric(all_metrics["row_b"], "psnr"),
            "row_c_psnr": average_metric(all_metrics["row_c"], "psnr"),
            "row_d_psnr": average_metric(all_metrics["row_d"], "psnr"),
            "row_b_ssim": average_metric(all_metrics["row_b"], "ssim"),
            "row_c_ssim": average_metric(all_metrics["row_c"], "ssim"),
            "row_d_ssim": average_metric(all_metrics["row_d"], "ssim"),
            "c_minus_b_psnr": average_metric(all_metrics["row_c"], "psnr") - average_metric(all_metrics["row_b"], "psnr"),
            "d_minus_c_psnr": average_metric(all_metrics["row_d"], "psnr") - average_metric(all_metrics["row_c"], "psnr"),
        }
    )
    save_csv(
        summary_rows,
        [
            "setting",
            "row_b_psnr",
            "row_c_psnr",
            "row_d_psnr",
            "row_b_ssim",
            "row_c_ssim",
            "row_d_ssim",
            "c_minus_b_psnr",
            "d_minus_c_psnr",
        ],
        TABLE_DIR / "summary_metrics.csv",
    )

    plot_grouped_metrics(all_metrics)
    plot_gain_bars(all_metrics)
    plot_loss_curves()
    selected = make_qualitative_figure()
    export_qualitative_cards(all_metrics, top_k=5)
    write_table_snippets(all_metrics, selected)

    print("Generated report assets in", REPORT_DIR)


if __name__ == "__main__":
    main()
