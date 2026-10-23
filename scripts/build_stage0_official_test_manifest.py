#!/usr/bin/env python3
"""Build a paired manifest for the official CDD-11 test split.

The official test layout contains one clear directory and eleven degraded
directories. Files are paired by filename with test/clear/<name>.  The
severity bins are intentionally unknown here; predicted-severity caching does
not use them.
"""
import argparse
import json
from pathlib import Path


CATEGORIES = [
    "clear",
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
TYPES = ["low", "haze", "rain", "snow"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Official CDD-11 test root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    clear_root = root / "clear"
    if not clear_root.is_dir():
        raise FileNotFoundError(clear_root)

    records = []
    counts = {}
    for category in CATEGORIES:
        lq_root = root / category
        if not lq_root.is_dir():
            raise FileNotFoundError(lq_root)
        files = sorted(p for p in lq_root.iterdir() if p.is_file())
        counts[category] = len(files)
        for lq in files:
            gt = clear_root / lq.name
            if not gt.is_file():
                raise FileNotFoundError(f"Missing clear target for {lq}")
            active = [] if category == "clear" else category.split("_")
            if any(name not in TYPES for name in active):
                raise ValueError(f"Unexpected category {category}")
            records.append({
                "id": f"official_test/{category}/{lq.name}",
                "split": "official_test",
                "scene_id": lq.stem,
                "lq_path": str(lq),
                "gt_path": str(gt),
                "severity_bins": [0, 0, 0, 0],
                "active_degradations": active,
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"records": len(records), "counts": counts, "output": str(output)}))


if __name__ == "__main__":
    main()
