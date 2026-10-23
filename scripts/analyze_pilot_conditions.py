"""Summarize cached Row-C condition probabilities by CDD-11 setting."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def entropy(probabilities):
    probabilities = np.clip(probabilities, 1e-8, 1 - 1e-8)
    return float(
        -np.mean(
            probabilities * np.log(probabilities)
            + (1 - probabilities) * np.log(1 - probabilities)
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.manifest) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("Empty condition manifest")

    grouped = defaultdict(list)
    for record in records:
        probabilities = np.asarray(record["rowc_probabilities"], dtype=np.float32)
        if probabilities.shape != (4,):
            raise ValueError(f"Expected four Row-C probabilities in {record['id']}")
        grouped[record.get("setting", "unknown")].append(probabilities)

    rows = []
    for setting, values in sorted(grouped.items()):
        matrix = np.stack(values)
        rows.append(
            {
                "setting": setting,
                "count": len(values),
                "mean_low": float(matrix[:, 0].mean()),
                "mean_haze": float(matrix[:, 1].mean()),
                "mean_rain": float(matrix[:, 2].mean()),
                "mean_snow": float(matrix[:, 3].mean()),
                "mean_binary_entropy": entropy(matrix),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"records": len(records), "settings": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
