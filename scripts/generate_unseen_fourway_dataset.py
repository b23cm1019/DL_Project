#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "report" / "unseen_4way" / "data"
OUT_DIR = OUT_ROOT / "low_haze_rain_snow"
META_PATH = ROOT / "report" / "unseen_4way" / "fourway_generation_notes.txt"

CDD_TEST = Path("/data/datasets/CDD-11_test")
LOW_HAZE = CDD_TEST / "low_haze"
LOW_HAZE_RAIN = CDD_TEST / "low_haze_rain"
LOW_HAZE_SNOW = CDD_TEST / "low_haze_snow"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_ids = sorted(p.stem for p in LOW_HAZE.glob("*.png"))
    for image_id in image_ids:
        low_haze = cv2.imread(str(LOW_HAZE / f"{image_id}.png"), cv2.IMREAD_COLOR).astype(np.int16)
        low_haze_rain = cv2.imread(str(LOW_HAZE_RAIN / f"{image_id}.png"), cv2.IMREAD_COLOR).astype(np.int16)
        low_haze_snow = cv2.imread(str(LOW_HAZE_SNOW / f"{image_id}.png"), cv2.IMREAD_COLOR).astype(np.int16)
        synthetic = np.clip(low_haze_rain + low_haze_snow - low_haze, 0, 255).astype(np.uint8)
        cv2.imwrite(str(OUT_DIR / f"{image_id}.png"), synthetic)

    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(
        "Synthetic unseen 4-way degradation folder: low_haze_rain_snow\n"
        "Generation rule per image: clip(low_haze_rain + low_haze_snow - low_haze, 0, 255)\n"
        "Interpretation: rain and snow overlays are composed on top of the shared low+haze base.\n"
        "This is an out-of-distribution qualitative stress test, not an official CDD-11 benchmark split.\n"
    )
    print(f"Generated {len(image_ids)} synthetic 4-way images in {OUT_DIR}")


if __name__ == "__main__":
    main()
