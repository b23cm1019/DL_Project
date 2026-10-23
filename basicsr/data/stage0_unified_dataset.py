"""Datasets for unified Stage 0 synthetic and CDD-11 supervision."""
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data

from basicsr.data.transforms import augment
from basicsr.utils.registry import DATASET_REGISTRY

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _read_manifest(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


@DATASET_REGISTRY.register()
class Stage0UnifiedDataset(data.Dataset):
    """Return both possible labels, with masks selecting valid supervision.

    ``mode=synthetic`` supplies severity labels and ``mode=cdd11`` supplies
    only type-presence labels. This makes mixed fine-tuning possible without
    ever pretending that CDD-11 has severity annotations.
    """

    def __init__(self, opt):
        self.opt = opt
        self.types = list(opt.get("degradation_types", ["low", "haze", "rain", "snow"]))
        self.crop_size = int(opt.get("crop_size", 224))
        self.mode = opt.get("mode", "synthetic")
        if self.mode == "synthetic":
            self.items = _read_manifest(opt["manifest"])
            self.samples = [(item["image_path"], self._severity(item)) for item in self.items]
        elif self.mode == "cdd11":
            root = Path(opt["dataroot_lq"])
            self.samples = []
            for category in opt["categories"]:
                folder = root / category
                if not folder.is_dir():
                    raise FileNotFoundError(f"Missing CDD-11 category folder: {folder}")
                labels = [float(name in category.split("_")) for name in self.types]
                self.samples.extend((str(path), labels) for path in sorted(folder.rglob("*"))
                                    if path.suffix.lower() in _IMAGE_EXTENSIONS)
        else:
            raise ValueError(f"Unsupported Stage 0 mode: {self.mode}")
        if not self.samples:
            raise ValueError("Stage 0 dataset is empty")
        # Keep all degradation variants of the same scene in one split.
        # This is used for CDD-11 development validation; the official test
        # root is intentionally not needed by the new Stage 0 configs.
        fraction = opt.get("scene_val_fraction")
        split_manifest = opt.get("scene_split_manifest")
        if self.mode == "cdd11" and split_manifest:
            split = json.loads(Path(split_manifest).read_text())
            key = "val_scene_ids" if opt.get("phase") == "val" else "train_scene_ids"
            allowed = set(split[key])
            self.samples = [(path, labels) for path, labels in self.samples if Path(path).stem in allowed]
            if not self.samples:
                raise ValueError(f"CDD-11 scene split produced an empty {key} dataset")
        elif self.mode == "cdd11" and fraction is not None:
            seed = str(opt.get("scene_split_seed", 0)).encode()
            val_ids = set()
            for image_path, _ in self.samples:
                scene_id = Path(image_path).stem
                digest = hashlib.md5(seed + scene_id.encode()).hexdigest()
                if int(digest[:8], 16) / 0xFFFFFFFF < float(fraction):
                    val_ids.add(scene_id)
            is_val = opt.get("phase") == "val"
            self.samples = [(path, labels) for path, labels in self.samples
                            if (Path(path).stem in val_ids) == is_val]
            if not self.samples:
                raise ValueError("CDD-11 scene split produced an empty dataset")

    def _severity(self, item):
        values = item["severity_bins"]
        if isinstance(values, dict):
            values = [values[name] for name in self.types]
        values = [int(value) for value in values]
        if len(values) != len(self.types) or any(value not in range(4) for value in values):
            raise ValueError("severity_bins must contain one value in [0, 3] per type")
        return values

    def _soft_target(self, item, values):
        supplied = item.get("severity_soft") if isinstance(item, dict) else None
        if supplied is None:
            target = torch.zeros(len(values), 4, dtype=torch.float32)
            target[torch.arange(len(values)), torch.tensor(values)] = 1.0
            return target
        if isinstance(supplied, dict):
            supplied = [supplied[name] for name in self.types]
        target = torch.tensor(supplied, dtype=torch.float32)
        if tuple(target.shape) != (len(values), 4):
            raise ValueError("severity_soft must have shape [num_types, 4]")
        if not torch.allclose(target.sum(-1), torch.ones(len(values)), atol=1e-4):
            raise ValueError("Each severity_soft distribution must sum to one")
        return target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, values = self.samples[index]
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
            image = image[top:top + self.crop_size, left:left + self.crop_size]
            image = augment(image, self.opt.get("use_hflip", True), self.opt.get("use_rot", True))
        else:
            top, left = (h - self.crop_size) // 2, (w - self.crop_size) // 2
            image = image[top:top + self.crop_size, left:left + self.crop_size]
        if self.mode == "synthetic":
            severity = torch.tensor(values, dtype=torch.long)
            severity_soft = self._soft_target(
                self.items[index] if hasattr(self, "items") else {}, values)
            type_labels = (severity > 0).float()
            severity_mask, type_mask = 1.0, 0.0
        else:
            type_labels = torch.tensor(values, dtype=torch.float32)
            severity = torch.zeros(len(self.types), dtype=torch.long)
            severity_soft = torch.zeros(len(self.types), 4, dtype=torch.float32)
            severity_mask, type_mask = 0.0, 1.0
        return {
            "lq": torch.from_numpy(image.transpose(2, 0, 1)).float().contiguous(),
            "severity_bins": severity,
            "severity_soft": severity_soft,
            "type_labels": type_labels,
            "severity_mask": torch.tensor(severity_mask),
            "type_mask": torch.tensor(type_mask),
            "lq_path": image_path,
        }
