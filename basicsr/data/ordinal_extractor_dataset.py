"""Manifest-backed data set for ordinal degradation-extractor training."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data

from basicsr.data.transforms import augment
from basicsr.utils.registry import DATASET_REGISTRY


def _read_manifest(path):
    path = Path(path)
    with path.open() as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        if path.suffix == ".json":
            items = json.load(handle)
            return items["items"] if isinstance(items, dict) else items
        return list(csv.DictReader(handle))


@DATASET_REGISTRY.register()
class OrdinalExtractorDataset(data.Dataset):
    """Reads ``image_path`` and ``severity_bins`` from JSON/JSONL/CSV.

    ``severity_bins`` may be a list ordered as ``degradation_types`` or a
    mapping such as ``{"low": 2, "haze": 0, "rain": 1, "snow": 0}``.
    Labels are integer bins: 0=absent, 1=mild, 2=moderate, 3=severe.
    """

    def __init__(self, opt):
        self.opt = opt
        self.types = opt["degradation_types"]
        self.items = _read_manifest(opt["manifest"])
        if not self.items:
            raise ValueError(f"Manifest is empty: {opt['manifest']}")
        self.crop_size = opt.get("crop_size", 224)

    def _labels(self, item):
        bins = item["severity_bins"]
        if isinstance(bins, str):
            bins = json.loads(bins)
        if isinstance(bins, dict):
            return [int(bins[name]) for name in self.types]
        if len(bins) != len(self.types):
            raise ValueError("severity_bins length must match degradation_types")
        return [int(value) for value in bins]

    def __getitem__(self, index):
        item = self.items[index]
        image_path = item["image_path"]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = image.shape[:2]
        if min(h, w) < self.crop_size:
            scale = self.crop_size / min(h, w)
            image = cv2.resize(image, (round(w * scale), round(h * scale)))
            h, w = image.shape[:2]
        if self.opt["phase"] == "train":
            top = np.random.randint(0, h - self.crop_size + 1)
            left = np.random.randint(0, w - self.crop_size + 1)
            image = image[top : top + self.crop_size, left : left + self.crop_size]
            image = augment(image, self.opt.get("use_hflip", True), self.opt.get("use_rot", True))
        else:
            top, left = (h - self.crop_size) // 2, (w - self.crop_size) // 2
            image = image[top : top + self.crop_size, left : left + self.crop_size]
        return {
            "lq": torch.from_numpy(image.transpose(2, 0, 1)).float().contiguous(),
            "severity_bins": torch.tensor(self._labels(item), dtype=torch.long),
            "lq_path": image_path,
        }

    def __len__(self):
        return len(self.items)
