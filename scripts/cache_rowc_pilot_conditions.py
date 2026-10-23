"""Cache direct Row-C conditions and optional CLIP text-space conditions.

The Row-C classifier consumes decoder hooks from the frozen Row-C NAFNet.  This
script performs that teacher pass once per image, then writes condition .npy
files and manifests for conditioned restoration training.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor

from basicsr.archs import build_network
from basicsr.utils.options import yaml_load


PRIMITIVES = ["low light", "haze", "rain", "snow"]


def state(path, key):
    value = torch.load(path, map_location="cpu")
    if key in value:
        return value[key]
    return value.get("params", value)


def load_models(rowc_opt, net_path, dc_path, device):
    net_cfg = yaml_load(rowc_opt)
    net_g = build_network(net_cfg["network_g"]).to(device).eval()
    net_dc = build_network(net_cfg["network_dc"]).to(device).eval()
    net_g.load_state_dict(state(net_path, "params_ema"), strict=True)
    net_dc.load_state_dict(state(dc_path, "params"), strict=True)
    return net_cfg, net_g, net_dc


def make_hooks(net_g):
    outputs = []

    def hook(_, __, output):
        outputs.append(output[-1] if isinstance(output, tuple) else output)

    handles = []
    for name, module in net_g.named_modules():
        if "decoder" in name and name.count(".") == 1:
            handles.append(module.register_forward_hook(hook))
    return outputs, handles


def cache_text_bank(model_name, pretrained, device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    prompts = [f"an image with {kind}" for kind in PRIMITIVES]
    tokenizer = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        text = model.encode_text(tokenizer(prompts).to(device))
        text = F.normalize(text, dim=-1)
    return model, preprocess, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rowc-opt", default="options/cdd_experiments/pretrain_multilabel.yml")
    parser.add_argument("--rowc-test-opt", default="options/cdd_experiments/test_multilabel.yml")
    parser.add_argument("--rowc-checkpoint", required=True)
    parser.add_argument("--rowc-classifier-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--mode", choices=("direct", "clip"), required=True)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, net_g, net_dc = load_models(
        args.rowc_opt, args.rowc_checkpoint, args.rowc_classifier_checkpoint, device
    )
    window = math.lcm(
        cfg["network_g"].get("window_size", 16),
        2 ** len(cfg["network_g"].get("enc_blk_nums", [])),
    )
    clip_model = preprocess = text = None
    if args.mode == "clip":
        clip_model, preprocess, text = cache_text_bank(
            args.clip_model, args.clip_pretrained, device
        )

    output_root = Path(args.output_root)
    records = []
    with open(args.manifest) as handle:
        inputs = [json.loads(line) for line in handle if line.strip()]

    for index, record in enumerate(inputs, 1):
        pil_image = Image.open(record["lq_path"]).convert("RGB")
        image = to_tensor(pil_image).unsqueeze(0).to(device)
        height, width = image.shape[-2:]
        pad_h, pad_w = (-height) % window, (-width) % window
        padded = F.pad(image, (0, pad_w, 0, pad_h), "reflect") if pad_h or pad_w else image
        hooks, handles = make_hooks(net_g)
        with torch.no_grad():
            net_g(padded)
            logits = net_dc(padded, hooks[::-1])
            probabilities = torch.sigmoid(logits).squeeze(0)
            if args.mode == "direct":
                condition = probabilities
            else:
                # Use Row-C probabilities to select/weight frozen text
                # concepts. The restoration model receives this text-space
                # condition, not a CLIP vision embedding.
                condition = F.normalize(probabilities @ text, dim=0)
        for handle in handles:
            handle.remove()

        path = output_root / (record["id"].replace("/", "__") + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, condition.cpu().numpy().astype("float32"))
        record = dict(record)
        record["condition_path"] = str(path)
        record["condition_mode"] = args.mode
        record["rowc_probabilities"] = probabilities.cpu().tolist()
        records.append(record)
        if index % 100 == 0 or index == len(inputs):
            print(f"Cached {index}/{len(inputs)}", flush=True)

    output = Path(args.output_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
