"""Compare one-pass and recurrent two-pass restoration on every test image.

The second pass deliberately recomputes the degradation-classifier prompt from
the first restored image.  This answers the deployment question: what happens
when the actual pass-1 output is fed back to the full network?
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from basicsr.data import build_dataloader, build_dataset
from basicsr.metrics import calculate_metric
from basicsr.models import build_model
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.options import parse_options


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", required=True, help="Existing BasicSR test YAML")
    parser.add_argument("--output", required=True, help="Directory for CSV, summary, and images")
    return parser.parse_args()


def restore_once(model, image):
    """Run the model's normal padding -> prompt/classifier -> restoration path."""
    model.lq = image
    model.pre_test()
    model.test()
    model.post_test()
    return model.output.detach()


def metrics(result, gt, metric_options):
    result_np = result.clamp(0, 1).detach().cpu().numpy()
    gt_np = gt.clamp(0, 1).detach().cpu().numpy()
    return {
        name: float(calculate_metric({"img": result_np, "img2": gt_np}, metric_opt))
        for name, metric_opt in metric_options.items()
    }


def mean_rows(rows, prefix):
    values = [row[prefix] for row in rows]
    return float(np.mean(values))


def main():
    args = parse_args()
    # parse_options expects its own command line. Supply just the test YAML.
    import sys
    sys.argv = [sys.argv[0], "-opt", args.opt]
    root_path = str(Path(__file__).resolve().parents[1])
    opt, _ = parse_options(root_path, is_train=False)
    opt["dist"] = False

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    model = build_model(opt)
    metric_options = opt["val"]["metrics"]
    rows = []

    for _, dataset_opt in sorted(opt["datasets"].items()):
        setting = dataset_opt["name"]
        dataset = build_dataset(dataset_opt)
        loader = build_dataloader(
            dataset, dataset_opt, num_gpu=opt["num_gpu"], dist=False,
            sampler=None, seed=opt["manual_seed"],
        )
        print(f"[{setting}] {len(dataset)} images", flush=True)
        for index, data in enumerate(loader, 1):
            model.feed_data(data)
            degraded = model.lq
            gt = model.gt
            pass1 = restore_once(model, degraded)
            pass2 = restore_once(model, pass1)
            metric0 = metrics(degraded, gt, metric_options)
            metric1 = metrics(pass1, gt, metric_options)
            metric2 = metrics(pass2, gt, metric_options)

            filename = Path(data["lq_path"][0]).name
            image_dir = output_root / "images" / setting
            imwrite(tensor2img(degraded), str(image_dir / "degraded" / filename))
            imwrite(tensor2img(pass1), str(image_dir / "pass1" / filename))
            imwrite(tensor2img(pass2), str(image_dir / "pass2" / filename))
            row = {"setting": setting, "filename": filename}
            for name in metric_options:
                row[f"degraded_{name}"] = metric0[name]
                row[f"pass1_{name}"] = metric1[name]
                row[f"pass2_{name}"] = metric2[name]
                row[f"pass2_minus_pass1_{name}"] = metric2[name] - metric1[name]
            rows.append(row)
            if index % 25 == 0 or index == len(dataset):
                print(f"[{setting}] {index}/{len(dataset)}", flush=True)
            del degraded, gt, pass1, pass2, model.output
            torch.cuda.empty_cache()

    fields = list(rows[0])
    with (output_root / "per_image_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"all": {"count": len(rows)}}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["setting"]].append(row)
    for setting, group in [("all", rows), *sorted(grouped.items())]:
        item = {"count": len(group)}
        for name in metric_options:
            for stage in ("degraded", "pass1", "pass2", "pass2_minus_pass1"):
                item[f"mean_{stage}_{name}"] = mean_rows(group, f"{stage}_{name}")
            item[f"pass2_better_{name}_count"] = sum(
                row[f"pass2_minus_pass1_{name}"] > 0 for row in group
            )
        summary[setting] = item
    with (output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["all"], indent=2), flush=True)


if __name__ == "__main__":
    main()
