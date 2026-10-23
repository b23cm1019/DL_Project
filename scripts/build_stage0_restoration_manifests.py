#!/usr/bin/env python3
"""Convert Stage-0 synthetic manifests into paired restoration manifests."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Stage-0 synthetic JSONL")
    parser.add_argument("--output", required=True, help="Restoration JSONL")
    args = parser.parse_args()

    records = []
    scenes = set()
    with Path(args.input).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            lq = Path(item["image_path"])
            gt = Path(item["source_image"])
            if not lq.is_file():
                raise FileNotFoundError(lq)
            if not gt.is_file():
                raise FileNotFoundError(gt)
            records.append({
                "id": f'{item["split"]}/{item["scene_id"]}/{lq.name}',
                "split": item["split"],
                "scene_id": item["scene_id"],
                "lq_path": str(lq),
                "gt_path": str(gt),
                "severity_bins": item["severity_bins"],
                "active_degradations": item.get("active_degradations", []),
            })
            scenes.add(item["scene_id"])
    if not records:
        raise ValueError(f"No records found in {args.input}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"records": len(records), "scenes": len(scenes), "output": str(output)}))


if __name__ == "__main__":
    main()
