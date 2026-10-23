#!/usr/bin/env python3
"""Cache oracle/predicted Stage-0 conditions for restoration ablations.

Condition dimensions:
  R1 oracle_type:       4
  R2 predicted_type:    4
  R3 oracle_severity:  16 (four one-hot severity distributions)
  R4 predicted_severity:16 (four softmax distributions)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from basicsr.archs import build_network


TYPES = ["low", "haze", "rain", "snow"]


def load_state(path):
    payload = torch.load(path, map_location="cpu")
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(payload, dict) and key in payload:
            payload = payload[key]
            break
    return {k[7:] if k.startswith("module.") else k: v for k, v in payload.items()}


def image_tensor(path, device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    height, width = image.shape[:2]
    crop = 224
    if min(height, width) < crop:
        scale = crop / min(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
        height, width = image.shape[:2]
    top, left = (height - crop) // 2, (width - crop) // 2
    image = image[top:top + crop, left:left + crop]
    return torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)


def condition(logits, bins, mode):
    if mode == "oracle_type":
        return (torch.tensor(bins, dtype=torch.float32) > 0).numpy()
    if mode == "oracle_severity":
        result = torch.zeros(4, 4)
        result[torch.arange(4), torch.tensor(bins)] = 1.0
        return result.flatten().numpy()
    probabilities = logits.softmax(-1).squeeze(0).cpu()
    if mode == "predicted_type":
        return probabilities[:, 1:].sum(-1).numpy()
    if mode == "predicted_severity":
        return probabilities.flatten().numpy()
    raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--mode", choices=("oracle_type", "predicted_type", "oracle_severity", "predicted_severity"), required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    network = build_network({
        "type": "OrdinalDegradationExtractor",
        "num_types": 4,
        "num_bins": 4,
        "widths": [64, 128, 256, 384],
    }).to(device).eval()
    network.load_state_dict(load_state(args.checkpoint), strict=True)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    with open(args.manifest) as handle:
        inputs = [json.loads(line) for line in handle if line.strip()]
    for index, item in enumerate(inputs, 1):
        bins = item["severity_bins"]
        with torch.no_grad():
            logits = network(image_tensor(item["lq_path"], device))
        value = condition(logits, bins, args.mode).astype("float32")
        condition_path = output_root / (item["id"].replace("/", "__") + ".npy")
        np.save(condition_path, value)
        record = dict(item)
        record["condition_path"] = str(condition_path)
        record["condition_mode"] = args.mode
        record["condition_dim"] = int(value.size)
        records.append(record)
        if index % 100 == 0 or index == len(inputs):
            print(f"Cached {index}/{len(inputs)}", flush=True)

    output = Path(args.output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"mode": args.mode, "records": len(records), "output": str(output)}))


if __name__ == "__main__":
    main()
