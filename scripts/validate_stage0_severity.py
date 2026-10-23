"""Audit synthetic severity labels against recorded image-level proxies."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.manifest).open() if line.strip()]
    if not records:
        raise ValueError("Manifest is empty")
    types = ["low", "haze", "rain", "snow"]
    metrics = ["mean_luminance_change", "contrast_loss", "mean_absolute_change",
               "rain_mask_coverage", "rain_contribution_mean",
               "snow_mask_coverage", "snow_contribution_mean"]
    report = {"records": len(records), "types": {}}
    for type_index, name in enumerate(types):
        grouped = {metric: defaultdict(list) for metric in metrics}
        for record in records:
            bin_value = int(record["severity_bins"][type_index])
            proxy = record.get("severity_proxies", {}).get("by_degradation", {}).get(name) or {}
            for metric in metrics:
                if metric in proxy:
                    grouped[metric][bin_value].append(float(proxy[metric]))
        type_report = {"bin_counts": {}, "proxy_means": {}, "monotonicity": {}}
        for metric in metrics:
            means = []
            for bin_value in range(4):
                values = grouped[metric].get(bin_value, [])
                type_report["bin_counts"].setdefault(str(bin_value), len(values))
                means.append(sum(values) / len(values) if values else None)
            type_report["proxy_means"][metric] = means
            observed = [value for value in means if value is not None]
            increasing = all(a <= b for a, b in zip(observed, observed[1:]))
            decreasing = all(a >= b for a, b in zip(observed, observed[1:]))
            type_report["monotonicity"][metric] = {
                "increasing": increasing,
                "decreasing": decreasing,
                "passes_directional_check": increasing or decreasing,
            }
        report["types"][name] = type_report
    report["interpretation"] = (
        "These are rendering proxies, not ground-truth perceptual severity. "
        "Review proxy monotonicity and representative image grids before trusting hard bins."
    )
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"records": len(records), "report": str(output)}))


if __name__ == "__main__":
    main()
