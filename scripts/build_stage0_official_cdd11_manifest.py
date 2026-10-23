#!/usr/bin/env python3
"""Build an official CDD-11 paired restoration manifest.

The CDD-11 training split is scene-disjoint according to stage0_cdd_split.json.
This script keeps the eleven degraded categories and excludes the clear folder,
matching the paper-style CDD-11 restoration evaluation.
"""
import argparse
import json
from pathlib import Path


CATEGORIES = [
    "low", "haze", "rain", "snow", "low_haze", "low_rain", "low_snow",
    "haze_rain", "haze_snow", "low_haze_rain", "low_haze_snow",
]
TYPES = {"low", "haze", "rain", "snow"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="CDD-11 train root")
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    clear_root = root / "clear"
    split = json.loads(Path(args.split_json).read_text())
    scene_ids = set(split[f"{args.split}_scene_ids"])
    records = []

    for category in CATEGORIES:
        category_root = root / category
        if not category_root.is_dir():
            raise FileNotFoundError(category_root)
        active = category.split("_")
        if not set(active).issubset(TYPES):
            raise ValueError(category)
        for lq in sorted(p for p in category_root.iterdir() if p.is_file()):
            scene_id = lq.stem
            if scene_id not in scene_ids:
                continue
            gt = clear_root / lq.name
            if not gt.is_file():
                raise FileNotFoundError(gt)
            records.append({
                "id": f"{args.split}/{category}/{lq.name}",
                "split": args.split,
                "setting": category,
                "scene_id": scene_id,
                "lq_path": str(lq),
                "gt_path": str(gt),
                "severity_bins": [0, 0, 0, 0],
                "active_degradations": active,
            })

    if not records:
        raise ValueError(f"No records found for split {args.split}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"split": args.split, "records": len(records),
                      "scenes": len(scene_ids), "output": str(output)}))


if __name__ == "__main__":
    main()
