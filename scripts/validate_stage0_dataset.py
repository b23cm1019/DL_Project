"""Check Stage 0 manifests for coverage, balance, and scene leakage."""
import argparse
import os
import json
from collections import Counter
from pathlib import Path

TYPES = ("low", "haze", "rain", "snow")
EXPECTED = {
    "clear", "low", "haze", "rain", "snow", "low_haze", "low_rain",
    "low_snow", "haze_rain", "haze_snow", "low_haze_rain", "low_haze_snow",
}


def read(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", nargs="+", required=True,
                        help="Named manifests such as train=/path/train.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get(
            "STAGE0_DATASET_REPORT",
            "/scratch/p24cs0203/krish/stage0_synthetic/reports/stage0_dataset_integrity.json",
        )),
        help="JSON integrity report path (kept outside the repository by default).",
    )
    args = parser.parse_args()
    named = {value.split("=", 1)[0]: read(value.split("=", 1)[1]) for value in args.manifest}
    scene_sets = {name: {item["scene_id"] for item in items} for name, items in named.items()}
    overlaps = {}
    names = list(scene_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = sorted(scene_sets[left] & scene_sets[right])
            if shared:
                overlaps[f"{left}:{right}"] = shared[:10]
    report = {"splits": {}, "scene_overlaps": overlaps}
    for name, items in named.items():
        categories = Counter("_".join(item.get("active_degradations", [])) or "clear" for item in items)
        severity = {kind: Counter() for kind in TYPES}
        joint = Counter()
        for item in items:
            bins = item["severity_bins"]
            joint[tuple(bins)] += 1
            for kind, value in zip(TYPES, bins):
                severity[kind][str(value)] += 1
            if not Path(item["image_path"]).is_file():
                raise FileNotFoundError(item["image_path"])
        missing = sorted(EXPECTED - set(categories)) if name != "unseen" else []
        report["splits"][name] = {
            "records": len(items),
            "scenes": len(scene_sets[name]),
            "categories": dict(sorted(categories.items())),
            "severity_counts": {kind: dict(sorted(counts.items())) for kind, counts in severity.items()},
            "joint_severity_combinations": len(joint),
            "missing_expected_categories": missing,
        }
    if overlaps:
        raise ValueError(f"Scene leakage detected: {list(overlaps)}")
    if any(report["splits"][name]["missing_expected_categories"] for name in report["splits"] if name != "unseen"):
        raise ValueError("At least one ordinary split is missing a required category")
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
