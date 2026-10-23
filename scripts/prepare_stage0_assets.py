"""Prepare isolated Stage 0 assets from the available project data.

This preparation is deterministic and provenance-aware. CDD-11 clear images
are reused in place; illumination and depth maps are generated as explicit
image-derived proxies because full MegaDepth/LIME maps are not distributed
with OneRestore. Rain and snow masks come from the official OneRestore sample
asset directories and are intentionally unaligned, as allowed by OneRestore.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--onerestore-assets", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=["proxy", "improved"], default="proxy")
    args = parser.parse_args()

    clean_root = Path(args.clean_root)
    source_root = Path(args.onerestore_assets)
    output_root = Path(args.output_root)
    light_root = output_root / "light_map"
    depth_root = output_root / "depth_map"
    rain_root = output_root / "rain_mask"
    snow_root = output_root / "snow_mask"
    for root in (light_root, depth_root, rain_root, snow_root):
        root.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(p for p in clean_root.rglob("*") if p.suffix.lower() in EXTENSIONS)
    rain_files = sorted((source_root / "rain_mask").glob("*"))
    snow_files = sorted((source_root / "snow_mask").glob("*"))
    if not clean_files or not rain_files or not snow_files:
        raise RuntimeError("Missing clean images or bundled OneRestore rain/snow samples")

    depth_model = None
    depth_transform = None
    depth_device = None
    if args.mode == "improved":
        import torch
        depth_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(depth_device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        depth_transform = transforms.small_transform

    for source in clean_files:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable clean image: {source}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # LIME-style illumination estimate from the maximum RGB channel.
        illumination = np.max(image.astype(np.float32) / 255.0, axis=2)
        illumination = cv2.blur(illumination, (15, 15))
        illumination = np.clip(illumination, 1e-3, 1.0)
        if args.mode == "improved":
            import torch
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sample = depth_transform(rgb).to(depth_device)
            with torch.no_grad():
                prediction = depth_model(sample)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=image.shape[:2], mode="bicubic",
                    align_corners=False).squeeze()
            depth = prediction.detach().cpu().numpy()
            depth = (depth - depth.min()) / max(float(depth.max() - depth.min()), 1e-6)
        else:
            # Deterministic proxy retained only for the pilot.
            depth = 1.0 - gray
            depth = cv2.GaussianBlur(depth, (0, 0), 1.2)
        cv2.imwrite(str(light_root / source.name), (illumination * 255).astype(np.uint8))
        cv2.imwrite(str(depth_root / source.name), (depth * 255).astype(np.uint8))

    for index, source in enumerate(rain_files):
        shutil.copy2(source, rain_root / f"rain_{index:05d}{source.suffix.lower()}")
    for index, source in enumerate(snow_files):
        shutil.copy2(source, snow_root / f"snow_{index:05d}{source.suffix.lower()}")

    metadata = {
        "clean_root": str(clean_root),
        "clean_images": len(clean_files),
        "mode": args.mode,
        "light_map": {"source": "LIME-style maximum-channel illumination estimate"},
        "depth_map": {
            "source": "MiDaS_small monocular depth estimate" if args.mode == "improved"
                      else "deterministic inverted-luminance depth proxy",
            "not_official_megadepth_output": args.mode != "improved",
        },
        "rain_mask_source": str(source_root / "rain_mask"),
        "snow_mask_source": str(source_root / "snow_mask"),
        "rain_masks": len(rain_files),
        "snow_masks": len(snow_files),
        "upstream_reference": "third_party/OneRestore/syn_data/data",
    }
    (output_root / "asset_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
