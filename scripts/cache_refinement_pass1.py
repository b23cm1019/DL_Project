"""Cache frozen Row-C pass-1 images with resume support."""
import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from basicsr.archs import build_network
from basicsr.utils.options import yaml_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--test-opt", default="options/cdd_experiments/test_multilabel.yml")
    parser.add_argument("--format", choices=("png", "jpg"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    options = yaml_load(args.test_opt)
    network = build_network(options["network_g"]).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cpu")
    network.load_state_dict(state.get("params_ema", state.get("params", state)), strict=True)
    window = math.lcm(
        options["network_g"].get("window_size", 16),
        2 ** len(options["network_g"].get("enc_blk_nums", [])),
    )

    with open(args.manifest) as handle:
        input_records = [json.loads(line) for line in handle if line.strip()]
    records = []
    for index, record in enumerate(input_records, 1):
        source_name = Path(record["lq_path"]).name
        source_path = Path(args.output_root) / record["split"] / record["setting"] / source_name
        output_path = source_path.with_suffix(".jpg" if args.format == "jpg" else ".png")
        legacy_png = source_path.with_suffix(".png")

        # Reuse outputs from this run or from the failed PNG run.
        if output_path.is_file() and output_path.stat().st_size > 0:
            record["pass1_path"] = str(output_path)
            records.append(record)
            continue
        if legacy_png.is_file() and legacy_png.stat().st_size > 0:
            record["pass1_path"] = str(legacy_png)
            records.append(record)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image = to_tensor(Image.open(record["lq_path"]).convert("RGB")).unsqueeze(0).cuda()
        height, width = image.shape[-2:]
        pad_h, pad_w = (-height) % window, (-width) % window
        padded = F.pad(image, (0, pad_w, 0, pad_h), "reflect") if pad_h or pad_w else image
        with torch.no_grad():
            restored = network(padded)[..., :height, :width].clamp(0, 1)
        array = (restored[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        output_image = Image.fromarray(array)
        if args.format == "jpg":
            output_image.save(output_path, quality=args.jpeg_quality, optimize=False)
        else:
            output_image.save(output_path)
        record["pass1_path"] = str(output_path)
        records.append(record)
        if index % 100 == 0:
            print(f"Processed {index}/{len(input_records)}", flush=True)

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = output_manifest.with_name(output_manifest.name + f".tmp.{os.getpid()}")
    with temporary_manifest.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    os.replace(temporary_manifest, output_manifest)
    print(f"Cached/resumed {len(records)} pass-1 outputs into {output_manifest}", flush=True)


if __name__ == "__main__":
    main()
