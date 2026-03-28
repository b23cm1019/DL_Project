#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
import generate_report_assets as gra


ROOT = Path(__file__).resolve().parents[1]
DEGRADATION = "low_haze_rain_snow"
DATA_ROOT = ROOT / "report" / "unseen_4way" / "data"
CARD_ROOT = ROOT / "report" / "unseen_4way" / "cards"
METRIC_CSV = ROOT / "report" / "unseen_4way" / "unseen_4way_metrics.csv"
MANIFEST_CSV = ROOT / "report" / "unseen_4way" / "unseen_4way_card_manifest.csv"


def configure_globals() -> None:
    gra.LQ_ROOT = DATA_ROOT
    gra.GT_ROOT = Path("/data/datasets/CDD-11_test/clear")
    gra.VIS_ROOTS = {
        "row_b": ROOT / "results" / "Test_Unseen4Way_Row_B" / "visualization",
        "row_c": ROOT / "results" / "Test_Unseen4Way_Row_C" / "visualization",
        "row_d": ROOT / "results" / "Test_Unseen4Way_Row_D" / "visualization",
    }
    gra.SUFFIX = {"row_b": "row_b", "row_c": "row_c", "row_d": "row_d"}


def export_cards(top_k: int = 5) -> None:
    configure_globals()
    CARD_ROOT.mkdir(parents=True, exist_ok=True)
    comparisons = [("row_c", "row_b"), ("row_d", "row_c"), ("row_d", "row_b")]
    manifest_rows: list[dict[str, object]] = []

    ids = sorted(
        p.name[: -len("_row_b.png")]
        for p in (gra.VIS_ROOTS["row_b"] / DEGRADATION).glob("*_row_b.png")
    )

    metric_rows = []
    for image_id in ids:
        row = {"image_id": image_id}
        for model in ("row_b", "row_c", "row_d"):
            row[f"{model}_psnr"] = round(gra.image_psnr(DEGRADATION, image_id, model), 4)
        row["c_minus_b"] = round(row["row_c_psnr"] - row["row_b_psnr"], 4)
        row["d_minus_c"] = round(row["row_d_psnr"] - row["row_c_psnr"], 4)
        row["d_minus_b"] = round(row["row_d_psnr"] - row["row_b_psnr"], 4)
        metric_rows.append(row)
    gra.save_csv(
        metric_rows,
        ["image_id", "row_b_psnr", "row_c_psnr", "row_d_psnr", "c_minus_b", "d_minus_c", "d_minus_b"],
        METRIC_CSV,
    )

    for challenger, baseline in comparisons:
        comparison_name = f"{challenger}_vs_{baseline}"
        out_dir = CARD_ROOT / comparison_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for existing in out_dir.glob("*.png"):
            existing.unlink()
        ranked = []
        for image_id in ids:
            gain = gra.image_psnr(DEGRADATION, image_id, challenger) - gra.image_psnr(DEGRADATION, image_id, baseline)
            ranked.append((image_id, gain))
        ranked.sort(key=lambda item: item[1], reverse=True)
        for rank, (image_id, gain) in enumerate(ranked[:top_k], start=1):
            card = gra.build_example_card(DEGRADATION, image_id, baseline=baseline, challenger=challenger, standalone=True)
            filename = f"{rank:02d}_{image_id}_{challenger}_minus_{baseline}.png"
            card.save(out_dir / filename)
            manifest_rows.append(
                {
                    "comparison": comparison_name,
                    "rank": rank,
                    "image_id": image_id,
                    "gain_db": round(gain, 4),
                    "row_b_psnr": round(gra.image_psnr(DEGRADATION, image_id, "row_b"), 4),
                    "row_c_psnr": round(gra.image_psnr(DEGRADATION, image_id, "row_c"), 4),
                    "row_d_psnr": round(gra.image_psnr(DEGRADATION, image_id, "row_d"), 4),
                    "file": filename,
                }
            )

    gra.save_csv(
        manifest_rows,
        ["comparison", "rank", "image_id", "gain_db", "row_b_psnr", "row_c_psnr", "row_d_psnr", "file"],
        MANIFEST_CSV,
    )
    print(f"Exported unseen 4-way cards to {CARD_ROOT}")


if __name__ == "__main__":
    export_cards(top_k=5)
