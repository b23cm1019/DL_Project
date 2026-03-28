#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
OUT = REPORT_DIR / "report_export.pdf"


def load_summary():
    rows = []
    with (TABLE_DIR / "summary_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def add_text_page(pdf: PdfPages, title: str, body_lines: list[str], footer: str | None = None):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.95, title, fontsize=18, fontweight="bold", va="top")
    y = 0.91
    for line in body_lines:
        if line == "":
            y -= 0.018
            continue
        style = dict(fontsize=10, va="top")
        if line.startswith("## "):
            fig.text(0.06, y, line[3:], fontsize=13, fontweight="bold", va="top")
            y -= 0.025
        elif line.startswith("- "):
            fig.text(0.08, y, u"\u2022 " + line[2:], **style)
            y -= 0.020
        else:
            fig.text(0.06, y, line, **style)
            y -= 0.020
    if footer:
        fig.text(0.06, 0.03, footer, fontsize=8, color="dimgray")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_paths: list[Path], captions: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.95, title, fontsize=18, fontweight="bold", va="top")
    top = 0.90
    slot_h = 0.38 if len(image_paths) == 2 else 0.72
    gap = 0.05
    for idx, (path, caption) in enumerate(zip(image_paths, captions)):
        y0 = top - idx * (slot_h + gap) - slot_h
        ax = fig.add_axes([0.08, y0, 0.84, slot_h - 0.05])
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        fig.text(0.08, y0 - 0.02, caption, fontsize=9, va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, summary_rows: list[dict[str, str]]):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.95, title, fontsize=18, fontweight="bold", va="top")

    avg = next(r for r in summary_rows if r["setting"] == "average")
    fig.text(
        0.06,
        0.90,
        "Average metrics: Row B = 24.4769 / 0.8045, Row C = 24.7653 / 0.8155, Row D = 24.8225 / 0.8136.",
        fontsize=10,
        va="top",
    )
    fig.text(
        0.06,
        0.875,
        "Row C improves over Row B on all 11 settings. Row D improves average PSNR further, but not average SSIM.",
        fontsize=10,
        va="top",
    )

    ax = fig.add_axes([0.05, 0.08, 0.90, 0.76])
    ax.axis("off")
    col_labels = ["Setting", "Row B", "Row C", "Row D", "C-B", "D-C"]
    table_rows = []
    for row in summary_rows:
        setting = row["setting"]
        if setting == "average":
            setting = "Average"
        table_rows.append(
            [
                setting,
                f"{float(row['row_b_psnr']):.3f} / {float(row['row_b_ssim']):.3f}",
                f"{float(row['row_c_psnr']):.3f} / {float(row['row_c_ssim']):.3f}",
                f"{float(row['row_d_psnr']):.3f} / {float(row['row_d_ssim']):.3f}",
                f"{float(row['c_minus_b_psnr']):+.3f}",
                f"{float(row['d_minus_c_psnr']):+.3f}",
            ]
        )
    table = ax.table(cellText=table_rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    summary_rows = load_summary()
    with PdfPages(OUT) as pdf:
        add_text_page(
            pdf,
            "Weak-Supervision Enhancements to DCPT",
            [
                "GitHub: https://github.com/b23cm1019/DL_Project",
                "YouTube: add final demo link before submission",
                "",
                "## Base Paper",
                "DCPT uses degradation classification as weak supervision for universal image restoration pretraining.",
                "The original pipeline first learns restoration features, then trains a lightweight classifier on those features, and finally finetunes the restoration model.",
                "",
                "## Our Contributions",
                "- Row B: local reduced-compute baseline with single-label degradation supervision.",
                "- Row C: replace 11-way single-label supervision with a 4D multi-label BCE target over primitive degradations.",
                "- Row D: reuse the frozen Row C classifier output as a semantic soft prompt during finetuning.",
                "",
                "## Reduced-Compute Setup",
                "- Dataset: CDD-11, balanced across all 11 test settings.",
                "- Architecture: NAFNet only.",
                "- Schedule: 25k pretraining + 100k finetuning iterations.",
                "- Budget relative to paper setup: about 14.7% of the original total iterations.",
                "",
                "## Key Findings",
                "- Row C improves over Row B on all 11 settings (+0.2884 dB, +0.0110 SSIM on average).",
                "- Row D attains the best average PSNR, but not the best average SSIM.",
                "- Multi-label BCE is the most reliable contribution under reduced compute; prompt injection is promising but selective.",
            ],
            footer="LaTeX source is provided separately in report/report.tex.",
        )

        add_table_page(pdf, "Quantitative Results", summary_rows)

        add_image_page(
            pdf,
            "Metric and Gain Visualizations",
            [FIG_DIR / "grouped_metrics.png", FIG_DIR / "psnr_gains.png"],
            [
                "Per-degradation PSNR/SSIM comparison. Row C consistently outperforms Row B across all 11 settings; Row D gives selective PSNR gains, especially on haze-heavy cases.",
                "PSNR gain plots. Top: Row C minus Row B. Bottom: Row D minus Row C.",
            ],
        )

        add_image_page(
            pdf,
            "Loss Curves and Qualitative Examples",
            [FIG_DIR / "pretrain_losses.png", FIG_DIR / "finetune_losses.png"],
            [
                "Pretraining losses for Row B and Row C. The classification-loss scales are not directly comparable because the objectives differ, but both stabilize.",
                "Finetuning pixel-loss curves for Row B, Row C, and Row D.",
            ],
        )

        add_image_page(
            pdf,
            "Qualitative Comparison and Insights",
            [FIG_DIR / "qualitative_examples.png"],
            [
                "Top row: haze example selected by maximum Row C - Row B PSNR gain. Bottom row: low_haze_rain example selected by maximum Row D - Row C PSNR gain among saved Row D visualizations.",
            ],
        )

    print(OUT)


if __name__ == "__main__":
    main()
