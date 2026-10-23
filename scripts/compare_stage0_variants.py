"""Verify that proxy and improved manifests use the same protocol."""
import argparse
import json
from pathlib import Path


def load(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--improved", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    proxy, improved = load(args.proxy), load(args.improved)
    if len(proxy) != len(improved):
        raise RuntimeError(f"Manifest length mismatch: {len(proxy)} != {len(improved)}")
    mismatches = []
    for index, (left, right) in enumerate(zip(proxy, improved)):
        keys = ("source_image", "scene_id", "split", "severity_bins",
                "active_degradations", "rendering_parameters")
        for key in keys:
            if left.get(key) != right.get(key):
                mismatches.append({"index": index, "key": key,
                                   "proxy": left.get(key), "improved": right.get(key)})
    report = {"proxy_records": len(proxy), "improved_records": len(improved),
              "protocol_match": not mismatches, "mismatches": mismatches[:100]}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if mismatches:
        raise RuntimeError("Proxy and improved manifests do not share the frozen protocol")


if __name__ == "__main__":
    main()
