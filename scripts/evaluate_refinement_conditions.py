"""Evaluate M0--M4 from cached pass-1 images and one M2 adapter checkpoint.

M0 is the original degraded input, M1 is a frozen Row-C second pass, and
M2/M3/M4 use the trained CLIP-feedback adapter with correct, deranged, and
oracle feedback respectively.  M3 and M4 deliberately reuse the M2 weights.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision.transforms.functional import to_tensor

from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.utils.options import yaml_load


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="train_clip/val_clip JSONL")
    p.add_argument("--m2-checkpoint", required=True, help="trained adapter checkpoint")
    p.add_argument("--rowc-checkpoint", required=True, help="frozen Row-C checkpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--network-opt", default="options/cdd_experiments/train_clip_feedback_refinement.yml")
    p.add_argument("--rowc-opt", default="options/cdd_experiments/test_multilabel.yml")
    p.add_argument("--window-size", type=int, default=16)
    p.add_argument("--shuffle-seed", type=int, default=2026)
    return p.parse_args()


def checkpoint_state(path):
    state = torch.load(path, map_location="cpu")
    for key in ("params_ema", "params", "state_dict"):
        if key in state:
            return state[key]
    return state


def load_net(opt_path, checkpoint, device):
    opt = yaml_load(opt_path)
    network = build_network(opt["network_g"])
    network.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return network.to(device).eval(), opt["network_g"]


def tensor_rgb(path, device):
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def restore(network, image, device, window_size):
    h, w = image.shape[-2:]
    pad_h, pad_w = (-h) % window_size, (-w) % window_size
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else image
    with torch.no_grad():
        output = network(padded)[..., :h, :w]
    return output.clamp(0, 1)


def restore_feedback(network, image, feedback, device, window_size):
    h, w = image.shape[-2:]
    pad_h, pad_w = (-h) % window_size, (-w) % window_size
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else image
    with torch.no_grad():
        output = network(padded, feedback[None].to(device))[..., :h, :w]
    return output.clamp(0, 1)


def metric_pair(image, gt):
    image = image.detach().cpu().numpy().transpose(0, 2, 3, 1)
    gt = gt.detach().cpu().numpy().transpose(0, 2, 3, 1)
    psnr = calculate_metric(
        {"img": image, "img2": gt},
        {"type": "calculate_psnr", "crop_border": 0, "test_y_channel": False, "image_range": 1.0},
    )
    # Use skimage here because this project’s legacy SSIM implementation
    # assumes 8/16-bit image conventions and can return NaN for [0, 1] data.
    ssim = structural_similarity(
        image[0], gt[0], data_range=1.0, channel_axis=2, gaussian_weights=True, sigma=1.5
    )
    return {"psnr": float(psnr), "ssim": float(ssim)}


def feedback_for(records, index, mode, permutation):
    if mode == "m4_oracle":
        return torch.tensor(records[index]["oracle_condition"], dtype=torch.float32)
    source = records[index] if mode == "m2_clip" else records[permutation[index]]
    return torch.from_numpy(np.load(source["clip_path"]).astype("float32"))


def main():
    a = args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(a.manifest) as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError("Empty manifest")
    required = {"lq_path", "gt_path", "pass1_path", "clip_path", "oracle_condition"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")

    rowc, rowc_cfg = load_net(a.rowc_opt, a.rowc_checkpoint, device)
    adapter, adapter_cfg = load_net(a.network_opt, a.m2_checkpoint, device)
    window = max(a.window_size, int(adapter_cfg.get("window_size", a.window_size)))
    if len(records) < 2:
        raise ValueError("M3 requires at least two manifest records")
    rng = np.random.default_rng(a.shuffle_seed)
    permutation = rng.permutation(len(records))
    while np.any(permutation == np.arange(len(records))):
        permutation = rng.permutation(len(records))
    rows = []
    for index, record in enumerate(records):
        degraded = tensor_rgb(record["lq_path"], device)
        gt = tensor_rgb(record["gt_path"], device)
        pass1 = tensor_rgb(record["pass1_path"], device)
        outputs = {
            "m0_degraded": degraded,
            "pass1": pass1,
            "m1_rowc": restore(rowc, pass1, device, window),
        }
        for name in ("m2_clip", "m3_shuffled_clip", "m4_oracle"):
            outputs[name] = restore_feedback(
                adapter, pass1, feedback_for(records, index, name, permutation), device, window
            )
        scores = {name: metric_pair(output, gt) for name, output in outputs.items()}
        row = {"id": record.get("id", str(index)), "setting": record.get("setting", "unknown")}
        for condition, values in scores.items():
            for metric, value in values.items():
                row[f"{condition}_{metric}"] = value
                if condition not in ("m0_degraded", "pass1"):
                    row[f"{condition}_delta_vs_pass1_{metric}"] = value - scores["pass1"][metric]
        rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"Evaluated {index + 1}/{len(records)}", flush=True)

    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "per_image_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["setting"]].append(row)
    grouped["all"] = rows
    summary = {}
    conditions = ("m0_degraded", "pass1", "m1_rowc", "m2_clip", "m3_shuffled_clip", "m4_oracle")
    for setting, group in sorted(grouped.items()):
        summary[setting] = {"count": len(group)}
        for condition in conditions:
            for metric in ("psnr", "ssim"):
                values = [row[f"{condition}_{metric}"] for row in group]
                summary[setting][f"mean_{condition}_{metric}"] = float(np.mean(values))
            if condition not in ("m0_degraded", "pass1"):
                for metric in ("psnr", "ssim"):
                    values = [row[f"{condition}_delta_vs_pass1_{metric}"] for row in group]
                    summary[setting][f"mean_{condition}_delta_{metric}"] = float(np.mean(values))
                    summary[setting][f"median_{condition}_delta_{metric}"] = float(np.median(values))
                    summary[setting][f"positive_{condition}_delta_{metric}_count"] = int(np.sum(np.asarray(values) > 0))
                    summary[setting][f"negative_{condition}_delta_{metric}_count"] = int(np.sum(np.asarray(values) < 0))
                    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                    summary[setting][f"ci95_halfwidth_{condition}_delta_{metric}"] = 1.96 * standard_error
        if setting == "all":
            summary[setting]["evaluation_note"] = (
                "This is the current development validation split. Use a genuinely held-out split for final paper claims."
            )
    with (output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary["all"], indent=2))


if __name__ == "__main__":
    main()
