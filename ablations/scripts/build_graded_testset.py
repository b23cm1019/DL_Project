"""Build the severity-graded CDD-11 test set (X1, and inputs for S3 / S9 / X2 / K6).

Clean images: CDD-11 *test* ground truth (never used for training).
For each clean image and each of the 11 CDD-11 compositions, render 4 severity levels with every
active primitive at the same level:
  levels 1-3  the midpoints of the training bins of the Stage 1 renderer
  level  4    beyond the training range (never rendered for training)
Masks, mask placement and the low-light noise pattern are fixed per (image, composition), so
within a composition only the severity changes from level to level.

The operators are the Stage 1 renderer's own (scripts/render_lsdir_cdd11_plan.py, imported, not
copied). Depth uses MiDaS exactly as scripts/prepare_lsdir_severity_assets.py; illumination is
max-RGB + Gaussian blur, also as there. Needs the MiDaS asset venv (see ablations/slurm/graded.sh).

Output: <out>/images/<composition>/L<level>/<name>.png and <out>/manifest.jsonl
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
COMPOSITIONS = (
    ("low",), ("haze",), ("rain",), ("snow",),
    ("low", "haze"), ("low", "rain"), ("low", "snow"),
    ("haze", "rain"), ("haze", "snow"),
    ("low", "haze", "rain"), ("low", "haze", "snow"),
)
TYPES = ("low", "haze", "rain", "snow")
# Bin ranges of scripts/render_lsdir_cdd11_plan.py::parameters (levels 1-3), plus level 4 beyond them.
RANGES = {
    "low": {1: (1.40, 1.80, .010, .030), 2: (1.80, 2.35, .030, .055), 3: (2.35, 3.00, .055, .080),
            4: (3.00, 3.60, .080, .100)},
    "haze": {1: (.55, .95, .60, .75), 2: (.95, 1.45, .65, .84), 3: (1.45, 2.00, .72, .90),
             4: (2.00, 2.60, .85, .95)},
    "overlay": {1: (.20, .40), 2: (.40, .70), 3: (.70, 1.00), 4: (1.00, 1.30)},
}


def load_renderer():
    spec = importlib.util.spec_from_file_location("stage1_renderer", ROOT / "scripts" / "render_lsdir_cdd11_plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def midpoint(low, high):
    return float((low + high) / 2)


def level_parameters(active, level):
    params = {}
    if "low" in active:
        r = RANGES["low"][level]
        params["low"] = {"gamma": midpoint(r[0], r[1]), "noise_std": midpoint(r[2], r[3])}
    if "haze" in active:
        r = RANGES["haze"][level]
        params["haze"] = {"beta": midpoint(r[0], r[1]), "atmospheric_light": midpoint(r[2], r[3])}
    for kind in ("rain", "snow"):
        if kind in active:
            params[kind] = {"intensity": midpoint(*RANGES["overlay"][level])}
    return params


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


def write_png(path, image_rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial" + path.suffix)
    bgr = cv2.cvtColor((np.clip(image_rgb, 0, 1) * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(partial), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"could not write {partial}")
    os.replace(partial, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test_root", default="/scratch/p26cs0003/krish/datasets/CDD11/test")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num_images", type=int, default=50, help="clean test images used (evenly spaced)")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--masks", type=Path,
                        default=Path("/scratch/p26cs0003/krish/severity_cdd11_lsdir_outdoor_v1/assets"),
                        help="folder with the Stage 1 rain_mask/ and snow_mask/ folders")
    parser.add_argument("--midas_model", default="MiDaS_small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()

    renderer = load_renderer()
    import torch
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    hub_dir = Path(torch.hub.get_dir()) / "intel-isl_MiDaS_master"
    source = ("local", str(hub_dir)) if hub_dir.is_dir() else ("github", "intel-isl/MiDaS")
    model = torch.hub.load(source[1], args.midas_model, source=source[0]).to(device).eval()
    transforms = torch.hub.load(source[1], "transforms", source=source[0])
    transform = getattr(transforms, "dpt_transform" if args.midas_model.startswith("DPT") else "small_transform")

    rain_masks = sorted((args.masks / "rain_mask").glob("*"))
    snow_masks = sorted((args.masks / "snow_mask").glob("*"))
    if not rain_masks or not snow_masks:
        raise FileNotFoundError(f"rain_mask/ and snow_mask/ not found under {args.masks}")
    clean_paths = sorted((Path(args.test_root) / "clear").glob("*.png"))
    step = max(1, len(clean_paths) // args.num_images)
    clean_paths = clean_paths[::step][:args.num_images]
    if any(level not in (1, 2, 3, 4) for level in args.levels):
        parser.error("levels must be in 1..4")

    rows = []
    for image_index, clean_path in enumerate(clean_paths):
        clean = renderer.rgb(clean_path)
        height, width = clean.shape[:2]
        light = cv2.GaussianBlur(np.max(clean, axis=2), (0, 0), 3.0)
        with torch.no_grad():
            sample = transform((clean * 255).round().astype(np.uint8)).to(device)
            depth = model(sample)
            depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=(height, width), mode="bicubic",
                                                    align_corners=False).squeeze().cpu().numpy()
        depth = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-6)
        # Round-trip through uint8 like the Stage 1 assets (PNG maps).
        light = np.round(np.clip(light, 0, 1) * 255) / 255.0
        depth = np.round(np.clip(depth, 0, 1) * 255) / 255.0
        light, depth = light.astype(np.float32), depth.astype(np.float32)
        for active in COMPOSITIONS:
            composition = "_".join(active)
            mask_rng = np.random.default_rng(seed_of(args.seed, clean_path.name, composition, "masks"))
            rain = renderer.transformed_mask(rain_masks[int(mask_rng.integers(len(rain_masks)))], height, width, mask_rng)
            snow = renderer.transformed_mask(snow_masks[int(mask_rng.integers(len(snow_masks)))], height, width, mask_rng)
            noise_seed = seed_of(args.seed, clean_path.name, composition, "noise")
            for level in args.levels:
                params = level_parameters(active, level)
                rendered = renderer.render(clean, light, depth, rain, snow, list(active), params,
                                           np.random.default_rng(noise_seed))
                out_path = args.out / "images" / composition / f"L{level}" / clean_path.name
                write_png(out_path, rendered)
                bins = [min(level, 3) if name in active else 0 for name in TYPES]
                rows.append({
                    "id": f"graded/{composition}/L{level}/{clean_path.name}", "composition": composition,
                    "level": level, "in_range": level <= 3, "severity_bins": bins,
                    "active_degradations": list(active), "lq_path": str(out_path.resolve()),
                    "gt_path": str(clean_path.resolve()), "rendering_parameters": params,
                })
        print(json.dumps({"rendered_clean_images": image_index + 1, "total": len(clean_paths)}), flush=True)

    manifest = args.out / "manifest.jsonl"
    with manifest.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    info = {"clean_images": len(clean_paths), "compositions": len(COMPOSITIONS), "levels": args.levels,
            "records": len(rows), "ranges": RANGES, "seed": args.seed, "depth": args.midas_model,
            "note": "levels 1-3 = training-bin midpoints; level 4 = beyond the training range "
                    "(severity_bins clamps it to 3, the closest trained bin)"}
    (args.out / "graded_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({"manifest": str(manifest), **info}, indent=2))


if __name__ == "__main__":
    main()
