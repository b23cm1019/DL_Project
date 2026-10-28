"""Manifest-backed paired dataset for paper-style DCPT pretraining.

The DCPT trainer expects paired ``lq``/``gt`` tensors.  The Stage-0 synthetic
generator instead writes JSONL records containing ``image_path`` (degraded),
``source_image`` (clean), and ``active_degradations``.  This adapter bridges
those formats without exposing severity labels to DCPT.

Each configured dataset filters one flat degradation category.  The enclosing
ConcatDataset then supplies the paper-style ``dataset_idx`` used by FocalLoss.
"""

import json
from pathlib import Path

import cv2
import torch
from torch.utils import data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.img_util import imfrombytes
from basicsr.utils.file_client import FileClient
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class DCPTManifestPairedDataset(data.Dataset):
    """Read synthetic JSONL records as paired DCPT samples."""

    def __init__(self, opt):
        self.opt = opt
        self.phase = opt.get("phase", "train")
        self.gt_size = opt.get("gt_size", 128)
        self.scale = opt.get("scale", 1)
        self.file_client = None
        self.io_backend_opt = dict(opt.get("io_backend", {"type": "disk"}))
        self.class_index = int(opt["class_index"])
        self.degradation_names = tuple(opt.get(
            "degradation_names",
            ("low", "haze", "rain", "snow"),
        ))

        manifest = Path(opt["manifest"])
        with manifest.open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        self.records = []
        for record in records:
            active = record.get("active_degradations")
            if active is None:
                raise ValueError(
                    f"Synthetic record lacks active_degradations: {record}"
                )
            active = tuple(sorted(active, key=self.degradation_names.index))
            if not active:
                # Clear samples are not one of the 11 DCPT degradation classes.
                continue
            key = "_".join(active)
            if key == opt["class_name"]:
                self.records.append(record)
        if not self.records:
            raise ValueError(
                f"No records for class {opt['class_name']} in {manifest}"
            )

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _read(path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype("float32") / 255.0

    def __getitem__(self, index):
        record = self.records[index]
        img_lq = self._read(record["image_path"])
        img_gt = self._read(record["source_image"])

        if self.phase == "train":
            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, self.gt_size, self.scale, record["image_path"]
            )
            img_gt, img_lq = augment(
                [img_gt, img_lq],
                self.opt.get("use_hflip", True),
                self.opt.get("use_rot", True),
            )

        img_gt = torch.from_numpy(img_gt.transpose(2, 0, 1)).float().contiguous()
        img_lq = torch.from_numpy(img_lq.transpose(2, 0, 1)).float().contiguous()
        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": record["image_path"],
            "gt_path": record["source_image"],
        }
