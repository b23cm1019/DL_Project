"""Validate and summarize a Stage 0 ordinal-label manifest without mutating it."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--types", nargs="+", required=True)
    args = parser.parse_args()
    counts = {name: Counter() for name in args.types}
    missing = []
    for index, item in enumerate(read(args.manifest), 1):
        if not Path(item["image_path"]).is_file(): missing.append(item["image_path"])
        bins = item["severity_bins"]
        if isinstance(bins, dict): bins = [bins[name] for name in args.types]
        if len(bins) != len(args.types) or any(int(value) not in range(4) for value in bins):
            raise ValueError(f"Invalid severity bins at record {index}")
        for name, value in zip(args.types, bins): counts[name][int(value)] += 1
    if missing: raise FileNotFoundError(f"{len(missing)} image paths are missing; first: {missing[0]}")
    print(json.dumps({"records": sum(counts[args.types[0]].values()), "per_type_bins": counts}, indent=2, default=dict))


if __name__ == "__main__": main()
