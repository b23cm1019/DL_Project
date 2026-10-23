"""Create CDD11 manifests for the one-step refinement proof-of-concept."""
import argparse
import json
from pathlib import Path

SETTINGS = ["low", "haze", "rain", "snow", "low_haze", "low_rain", "low_snow", "haze_rain", "haze_snow", "low_haze_rain", "low_haze_snow"]
TYPES = ["low", "haze", "rain", "snow"]


def oracle(setting):
    # CDD11 supplies type combinations, not actual mild/moderate/severe labels.
    result = []
    active = set(setting.split("_"))
    for kind in TYPES:
        result.extend([0., 1., 0., 0.] if kind in active else [1., 0., 0., 0.])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lq-root", required=True)
    parser.add_argument("--gt-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    args = parser.parse_args()
    records = []
    for setting in SETTINGS:
        for lq in sorted((Path(args.lq_root) / setting).glob("*")):
            gt = Path(args.gt_root) / lq.name
            if lq.is_file() and gt.is_file():
                records.append({"id": f"{args.split}/{setting}/{lq.name}", "split": args.split, "setting": setting, "lq_path": str(lq), "gt_path": str(gt), "oracle_condition": oracle(setting)})
    if not records: raise ValueError("No matched pairs found")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        for record in records: handle.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} samples to {args.output}")


if __name__ == "__main__": main()
