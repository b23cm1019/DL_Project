#!/usr/bin/env python3
"""Evaluate R0-R4 restoration checkpoints on the same paired manifest."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

from basicsr.archs import build_network
from basicsr.metrics import calculate_metric


def state(path):
    payload = torch.load(path, map_location="cpu")
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
            break
    return {k[7:] if k.startswith("module.") else k: v for k, v in payload.items()}


def read_image(path, device):
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(value.transpose(2, 0, 1)).float().unsqueeze(0).to(device)


def restore(net, lq, condition=None):
    height, width = lq.shape[-2:]
    multiple = 2 ** len(net.encoders)
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    padded = F.pad(lq, (0, pad_w, 0, pad_h), "reflect") if pad_h or pad_w else lq
    with torch.no_grad():
        output = net(padded) if condition is None else net(padded, condition)
    return output[..., :height, :width].clamp(0, 1)


def scores(output, target):
    result = output.cpu().numpy().transpose(0, 2, 3, 1)
    reference = target.cpu().numpy().transpose(0, 2, 3, 1)
    psnr = calculate_metric(
        {"img": result, "img2": reference},
        {"type": "calculate_psnr", "crop_border": 0, "test_y_channel": False, "image_range": 1.0},
    )
    ssim = structural_similarity(reference[0], result[0], data_range=1.0, channel_axis=2)
    return float(psnr), float(ssim)


def build_model(mode, checkpoint, condition_dim, device):
    common = {
        "width": 64,
        "enc_blk_nums": [1, 1, 1, 28],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
        "window_size": 16,
    }
    if mode == "R0":
        config = {"type": "NAFNet", **common}
    else:
        config = {
            "type": "CLIPFeedbackRefinementNAFNet",
            "feedback_dim": condition_dim,
            "adapter_dim": 128,
            **common,
        }
    network = build_network(config).to(device).eval()
    network.load_state_dict(state(checkpoint), strict=False)
    return network


def setting(record):
    active = record.get("active_degradations", [])
    return "_".join(active) if active else "clear"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("R0", "R1", "R2", "R3", "R4"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--condition-dim", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.manifest) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("Empty restoration manifest")
    if args.mode != "R0":
        for record in records:
            if "condition_path" not in record:
                raise ValueError(f"Missing condition_path for {args.mode}: {record}")

    device = torch.device(args.device)
    network = build_model(args.mode, args.checkpoint, args.condition_dim, device)
    rows = []
    for index, record in enumerate(records, 1):
        lq = read_image(record["lq_path"], device)
        gt = read_image(record["gt_path"], device)
        condition = None
        if args.mode != "R0":
            value = np.load(record["condition_path"]).astype("float32")
            if value.shape != (args.condition_dim,):
                raise ValueError(f"Condition shape {value.shape} != {(args.condition_dim,)}")
            condition = torch.from_numpy(value).unsqueeze(0).to(device)
        output = restore(network, lq, condition)
        psnr, ssim = scores(output, gt)
        rows.append({
            "id": record.get("id", str(index)),
            "scene_id": record.get("scene_id", "unknown"),
            "setting": setting(record),
            "psnr": psnr,
            "ssim": ssim,
        })
        if index % 100 == 0 or index == len(records):
            print(f"Evaluated {index}/{len(records)}", flush=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["setting"]].append(row)
    grouped["all"] = rows
    summary = {}
    for name, group in sorted(grouped.items()):
        summary[name] = {
            "count": len(group),
            "mean_psnr": float(np.mean([row["psnr"] for row in group])),
            "mean_ssim": float(np.mean([row["ssim"] for row in group])),
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"mode": args.mode, "summary": summary["all"]}, indent=2))


if __name__ == "__main__":
    main()
