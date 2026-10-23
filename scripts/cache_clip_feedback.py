"""Derive a 16-D CLIP degradation state from every cached pass-1 output."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROMPTS = [f"an image {level} {kind}" for kind in ["low light", "haze", "rain", "snow"] for level in ["without", "mild", "moderate", "severe"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    args = parser.parse_args()
    try: import open_clip
    except ImportError as error: raise RuntimeError("open_clip_torch is required in the GPU environment.") from error
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    with torch.no_grad():
        text = torch.nn.functional.normalize(model.encode_text(open_clip.get_tokenizer(args.model)(PROMPTS).to(device)), dim=-1)
    output_root, records = Path(args.output_root), []
    with open(args.manifest) as handle:
        for line in handle:
            record = json.loads(line)
            with torch.no_grad():
                image = preprocess(Image.open(record["pass1_path"]).convert("RGB")).unsqueeze(0).to(device)
                image = torch.nn.functional.normalize(model.encode_image(image), dim=-1)
                condition = (100 * image @ text.T).reshape(4, 4).softmax(1).flatten().cpu().numpy().astype("float32")
            path = output_root / (record["id"].replace("/", "__") + ".npy")
            path.parent.mkdir(parents=True, exist_ok=True); np.save(path, condition)
            record["clip_path"] = str(path); records.append(record)
    with open(args.output_manifest, "w") as handle:
        for record in records: handle.write(json.dumps(record) + "\n")


if __name__ == "__main__": main()
