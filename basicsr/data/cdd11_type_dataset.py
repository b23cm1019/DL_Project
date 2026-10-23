"""CDD-11 folder dataset for Stage 0a degradation-presence training."""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data

from basicsr.data.transforms import augment
from basicsr.utils.registry import DATASET_REGISTRY

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@DATASET_REGISTRY.register()
class CDD11TypeDataset(data.Dataset):
    """Read CDD-11 images and derive four primitive presence labels."""

    def __init__(self, opt):
        self.opt = opt
        self.root = Path(opt["dataroot_lq"])
        self.types = list(opt.get("degradation_types", ["low", "haze", "rain", "snow"]))
        self.crop_size = int(opt.get("crop_size", 224))
        self.samples = []
        for category in opt.get("categories", []):
            folder = self.root / category
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing CDD-11 category folder: {folder}")
            labels = [float(name in category.split("_")) for name in self.types]
            self.samples.extend((path, labels) for path in sorted(folder.rglob("*"))
                                if path.suffix.lower() in _IMAGE_EXTENSIONS)
        if not self.samples:
            raise ValueError(f"No images found below {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, labels = self.samples[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
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
        return {
            "lq": torch.from_numpy(image.transpose(2, 0, 1)).float().contiguous(),
            "type_labels": torch.tensor(labels, dtype=torch.float32),
            "lq_path": str(path),
        }
