"""Evaluate one E1-E5 conditioned pilot checkpoint on a cached manifest."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.utils.options import yaml_load


def image(path, device):
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(value.transpose(2, 0, 1)).float().unsqueeze(0).to(device)


def state(path):
    value = torch.load(path, map_location="cpu")
    for key in ("params_ema", "params", "state_dict"):
        if key in value:
            return value[key]
    return value


def restore(net, lq, condition):
    height, width = lq.shape[-2:]
    multiple = 2 ** len(net.encoders)
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    padded = F.pad(lq, (0, pad_w, 0, pad_h), "reflect") if pad_h or pad_w else lq
    with torch.no_grad():
        return net(padded, condition)[..., :height, :width].clamp(0, 1)


def score(output, gt):
    result = output.detach().cpu().numpy().transpose(0, 2, 3, 1)
    target = gt.detach().cpu().numpy().transpose(0, 2, 3, 1)
    psnr = calculate_metric(
        {"img": result, "img2": target},
        {"type": "calculate_psnr", "crop_border": 0, "test_y_channel": False, "image_range": 1.0},
    )
    from skimage.metrics import structural_similarity
    ssim = structural_similarity(result[0], target[0], data_range=1.0, channel_axis=2)
    return float(psnr), float(ssim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="options/cdd_experiments/pilot_conditioned.yml")
    parser.add_argument("--condition-mode", default="direct")
    parser.add_argument("--condition-dim", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml_load(args.config)
    network_config = dict(config["network_g"])
    network_config["feedback_dim"] = args.condition_dim
    net = build_network(network_config).to(device).eval()
    net.load_state_dict(state(args.checkpoint), strict=False)

    with open(args.manifest) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("Empty evaluation manifest")
    condition_matrix = np.stack(
        [np.load(record["condition_path"]).astype(np.float32) for record in records]
    )
    condition_mean = condition_matrix.mean(0)
    condition_std = condition_matrix.std(0).clip(min=1e-4)
    permutation = np.random.default_rng(2026).permutation(len(records))
    rows = []
    for index, record in enumerate(records, 1):
        lq = image(record["lq_path"], device)
        gt = image(record["gt_path"], device)
        condition = condition_matrix[index - 1].copy()
        if args.condition_mode == "zero":
            condition[:] = 0
        elif args.condition_mode == "shuffled":
            condition = condition_matrix[permutation[index - 1]].copy()
        elif args.condition_mode == "random":
            rng = np.random.default_rng(2026 + index - 1)
            condition = rng.normal(condition_mean, condition_std).astype(np.float32)
        condition = torch.from_numpy(condition).unsqueeze(0).to(device)
        output = restore(net, lq, condition)
        psnr, ssim = score(output, gt)
        rows.append({"id": record.get("id", str(index)), "setting": record.get("setting", "unknown"), "psnr": psnr, "ssim": ssim})
        if index % 100 == 0 or index == len(records):
            print(f"Evaluated {index}/{len(records)}", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["setting"]].append(row)
    grouped["all"] = rows
    summary = {}
    for setting, group in sorted(grouped.items()):
        summary[setting] = {
            "count": len(group),
            "mean_psnr": float(np.mean([row["psnr"] for row in group])),
            "mean_ssim": float(np.mean([row["ssim"] for row in group])),
        }
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary["all"], indent=2))


if __name__ == "__main__":
    main()
