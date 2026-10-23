"""Generate Stage 0 data with the released OneRestore-style operators.

Reference implementation vendored at ``third_party/OneRestore`` from
upstream commit ``23a2e6af066864dca013aeed7f6c6f5958aa7259``
(``syn_data/syn_data.py``). This file intentionally wraps and extends those
operators rather than importing the upstream script directly: Stage 0 needs
deterministic RNG, scene-level splits, severity metadata, soft labels, and
unseen-combination manifests.

The rendering operators follow OneRestore's ``syn_data.py``: illumination
map + guided filtering for low-light, additive rain masks, snow-mask blending,
and depth-dependent atmospheric scattering for haze. Severity control only
changes exposed strengths and is recorded separately from the class labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TYPES = ("low", "haze", "rain", "snow")
CDD11_COMBINATIONS = (
    (), ("low",), ("haze",), ("rain",), ("snow",),
    ("low", "haze"), ("low", "rain"), ("low", "snow"),
    ("haze", "rain"), ("haze", "snow"),
    ("low", "haze", "rain"), ("low", "haze", "snow"),
)


def guide_filter(guide, image, window=(3, 3), epsilon=0.01):
    mean_i = cv2.blur(guide, window)
    mean_p = cv2.blur(image, window)
    mean_ii = cv2.blur(guide * guide, window)
    mean_ip = cv2.blur(guide * image, window)
    variance = mean_ii - mean_i * mean_i
    covariance = mean_ip - mean_i * mean_p
    a = covariance / (variance + epsilon)
    b = mean_p - a * mean_i
    return cv2.blur(a, window) * guide + cv2.blur(b, window)


def syn_low(image, light, image_gray, gamma, noise_std, rng):
    light = guide_filter(light, image_gray)[:, :, None]
    reflectance = image / (light + 1e-7)
    illumination = (light + 1e-7) ** gamma
    noise = rng.normal(0, noise_std, image.shape).astype(np.float32)
    return np.clip(reflectance * illumination + noise, 0, 1)


def syn_haze(image, depth, beta, atmospheric_light):
    transmission = np.exp(-np.minimum(1 - cv2.blur(depth, (22, 22)), 0.7) * beta)
    return np.clip(image * transmission + atmospheric_light * (1 - transmission), 0, 1)


def render_one(clean, light, depth, rain_mask, snow_mask, active, params):
    image = clean.copy()
    if "low" in active:
        image = syn_low(image, light, cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY),
                        params["low"]["gamma"], params["low"]["noise_std"], params["rng"])
    if "rain" in active:
        image = np.clip(image + params["rain"]["intensity"] * rain_mask, 0, 1)
    if "snow" in active:
        strength = params["snow"]["intensity"]
        image = np.clip(image * (1 - strength * snow_mask) + strength * snow_mask, 0, 1)
    if "haze" in active:
        image = syn_haze(image, depth, params["haze"]["beta"], params["haze"]["atmospheric_light"])
    return image


def sample_parameters(bins, rng):
    # Ranges extend the original operators while retaining their parameterization.
    low_ranges = {1: (1.8, 2.1, 0.03, 0.04), 2: (2.1, 2.5, 0.04, 0.06), 3: (2.5, 3.0, 0.06, 0.08)}
    haze_ranges = {1: (0.8, 1.2), 2: (1.2, 1.6), 3: (1.6, 2.0)}
    rain_ranges = {1: (0.25, 0.45), 2: (0.45, 0.7), 3: (0.7, 1.0)}
    snow_ranges = {1: (0.25, 0.45), 2: (0.45, 0.7), 3: (0.7, 1.0)}
    low = low_ranges.get(bins[0])
    haze = haze_ranges.get(bins[1])
    return {
        "low": {"gamma": float(rng.uniform(low[0], low[1])) if low else None,
                 "noise_std": float(rng.uniform(low[2], low[3])) if low else None},
        "haze": {"beta": float(rng.uniform(haze[0], haze[1])) if haze else None,
                  "atmospheric_light": float(rng.uniform(0.6, 0.9)) if haze else None},
        "rain": {"intensity": float(rng.uniform(*rain_ranges[bins[2]])) if bins[2] else None},
        "snow": {"intensity": float(rng.uniform(*snow_ranges[bins[3]])) if bins[3] else None},
    }


def proxies(clean, rendered, rain_mask=None, snow_mask=None):
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY)
    image_gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)
    result = {
        "mean_luminance_change": float(image_gray.mean() - clean_gray.mean()),
        "contrast_loss": float(1 - image_gray.std() / max(float(clean_gray.std()), 1e-6)),
        "mean_absolute_change": float(np.abs(rendered - clean).mean()),
    }
    if rain_mask is not None:
        result["rain_mask_coverage"] = float((rain_mask.mean(axis=2) > 0.03).mean())
        result["rain_contribution_mean"] = float(rain_mask.mean())
    if snow_mask is not None:
        result["snow_mask_coverage"] = float((snow_mask.mean(axis=2) > 0.03).mean())
        result["snow_contribution_mean"] = float(snow_mask.mean())
    return result


def scene_split(files, split, val_fraction, test_fraction, unseen_fraction, seed):
    ranked = sorted(files, key=lambda p: hashlib.md5(f"{seed}:{p.stem}".encode()).hexdigest())
    train_end = max(1, round(len(ranked) * (1 - val_fraction - test_fraction - unseen_fraction)))
    val_end = min(len(ranked) - 3, train_end + round(len(ranked) * val_fraction))
    test_end = min(len(ranked) - 1, val_end + round(len(ranked) * test_fraction))
    if split == "train":
        return ranked[:train_end]
    if split == "val":
        return ranked[train_end:val_end]
    if split == "test":
        return ranked[val_end:test_end]
    return ranked[test_end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--light-root", required=True)
    parser.add_argument("--depth-root", required=True)
    parser.add_argument("--rain-root", required=True)
    parser.add_argument("--snow-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "unseen"], required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--unseen-fraction", type=float, default=0.1)
    parser.add_argument("--unseen-combination", default="rain_snow")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--samples-per-scene", type=int, default=12)
    parser.add_argument("--max-scenes", type=int, default=0,
                        help="Limit scenes for a pilot; 0 means all scenes.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    files = sorted(p for p in Path(args.clean_root).rglob("*") if p.suffix.lower() in EXTENSIONS)
    files = scene_split(files, args.split, args.val_fraction, args.test_fraction,
                        args.unseen_fraction, args.split_seed)
    if args.max_scenes > 0:
        files = files[:args.max_scenes]
    rain_files = sorted(p for p in Path(args.rain_root).rglob("*") if p.suffix.lower() in EXTENSIONS)
    snow_files = sorted(p for p in Path(args.snow_root).rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not files or not rain_files or not snow_files:
        raise ValueError("Clean, rain-mask, and snow-mask inputs must be non-empty")
    output_root = Path(args.output_root) / args.split
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for source in files:
        clean_bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        light = cv2.imread(str(Path(args.light_root) / source.name), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(Path(args.depth_root) / source.name), cv2.IMREAD_COLOR)
        if clean_bgr is None or light is None or depth is None:
            raise FileNotFoundError(f"Missing clean/light/depth asset for {source.name}")
        clean = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        height, width = clean.shape[:2]
        light = cv2.resize(light.astype(np.float32) / 255, (width, height))
        depth = cv2.resize(depth.astype(np.float32) / 255, (width, height))
        rain_path = rain_files[int(rng.integers(len(rain_files)))]
        snow_path = snow_files[int(rng.integers(len(snow_files)))]
        rain_mask = cv2.resize(cv2.imread(str(rain_path), cv2.IMREAD_COLOR).astype(np.float32) / 255, (width, height))
        snow_mask = cv2.resize(cv2.imread(str(snow_path), cv2.IMREAD_COLOR).astype(np.float32) / 255, (width, height))
        combinations = CDD11_COMBINATIONS
        if args.split == "unseen":
            combinations = (tuple(args.unseen_combination.split("_")),)
        for sample_index in range(args.samples_per_scene):
            # Cycle through categories so every sufficiently sized split
            # contains clear, single, pair, and triple degradations.
            active = combinations[sample_index % len(combinations)]
            bins = [int(rng.integers(1, 4)) if name in active else 0 for name in TYPES]
            params = sample_parameters(bins, rng)
            params["rng"] = rng
            rendered = render_one(clean, light, depth, rain_mask, snow_mask, active, params)
            isolated_proxies = {}
            for name in TYPES:
                isolated = render_one(clean, light, depth, rain_mask, snow_mask,
                                      (name,) if name in active else (), params)
                isolated_proxies[name] = proxies(
                    clean, isolated,
                    rain_mask if name == "rain" else None,
                    snow_mask if name == "snow" else None,
                ) if name in active else None
            output_path = output_root / f"{source.stem}_{sample_index:04d}.png"
            cv2.imwrite(str(output_path), cv2.cvtColor((rendered * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            records.append({
                "image_path": str(output_path), "source_image": str(source),
                "scene_id": source.stem, "split": args.split,
                "severity_bins": bins, "active_degradations": list(active),
                "rendering_parameters": {key: value for key, value in params.items() if key != "rng"},
                "asset_ids": {"light": source.name, "depth": source.name,
                              "rain_mask": rain_path.name, "snow_mask": snow_path.name},
                "severity_proxies": {
                    "final": proxies(clean, rendered, rain_mask, snow_mask),
                    "by_degradation": isolated_proxies,
                },
                "generator": "OneRestore syn_data.py operators with controlled strengths",
                "label_note": "Bins require proxy monotonicity and visual validation; parameters are not perceptual ground truth.",
            })
    with manifest.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"split": args.split, "records": len(records), "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
