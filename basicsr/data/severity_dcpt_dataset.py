"""Paired datasets for the severity-aware, paper-faithful DCPT pipeline."""
import json
from pathlib import Path

import cv2
import torch
from torch.utils import data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.registry import DATASET_REGISTRY


def _read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.0


@DATASET_REGISTRY.register()
class SeverityDCPTPairedDataset(data.Dataset):
    """Return paired images plus exact or censored degradation supervision.

    ``source=synthetic`` expects the generated manifest fields ``image_path``,
    ``source_image``, and ``severity_bins``.  It provides exact ordinal labels.
    ``source=cdd11`` expects the official CDD manifest fields ``lq_path``,
    ``gt_path``, and ``active_degradations``.  It provides only primitive
    presence labels; CDD-11 severity is never fabricated.
    """

    def __init__(self, opt):
        self.opt = opt
        self.phase = opt.get("phase", "train")
        self.source = opt["source"]
        if self.source not in {"synthetic", "cdd11"}:
            raise ValueError("source must be 'synthetic' or 'cdd11'")
        self.types = tuple(opt.get("degradation_types", ("low", "haze", "rain", "snow")))
        self.gt_size = int(opt.get("gt_size", 128))
        self.scale = int(opt.get("scale", 1))
        with Path(opt["manifest"]).open() as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Empty manifest: {opt['manifest']}")

    def __len__(self):
        return len(self.records)

    def _labels(self, record):
        if self.source == "synthetic":
            bins = record.get("severity_bins")
            if bins is None:
                raise ValueError(f"Synthetic record lacks severity_bins: {record}")
            if isinstance(bins, dict):
                bins = [bins[name] for name in self.types]
            bins = [int(value) for value in bins]
            if len(bins) != len(self.types) or any(value not in range(4) for value in bins):
                raise ValueError("severity_bins must contain four values in [0, 3]")
            return torch.tensor(bins, dtype=torch.long), torch.tensor(
                [float(value > 0) for value in bins], dtype=torch.float32
            ), True

        active = set(record.get("active_degradations", record.get("setting", "").split("_")))
        type_labels = torch.tensor([float(name in active) for name in self.types], dtype=torch.float32)
        # Placeholder only. Stage 2 masks ordinal loss for CDD-11 examples.
        return torch.zeros(len(self.types), dtype=torch.long), type_labels, False

    def __getitem__(self, index):
        record = self.records[index]
        lq_path = record.get("image_path", record.get("lq_path"))
        gt_path = record.get("source_image", record.get("gt_path"))
        if lq_path is None or gt_path is None:
            raise ValueError(f"Paired paths are missing from record: {record}")
        lq, gt = _read_rgb(lq_path), _read_rgb(gt_path)
        if self.phase == "train":
            gt, lq = paired_random_crop(gt, lq, self.gt_size, self.scale, lq_path)
            gt, lq = augment(
                [gt, lq], self.opt.get("use_hflip", True), self.opt.get("use_rot", True)
            )
        severity_bins, type_labels, has_severity = self._labels(record)
        return {
            "lq": torch.from_numpy(lq.transpose(2, 0, 1)).float().contiguous(),
            "gt": torch.from_numpy(gt.transpose(2, 0, 1)).float().contiguous(),
            "severity_bins": severity_bins,
            "type_labels": type_labels,
            "has_severity": torch.tensor(has_severity, dtype=torch.bool),
            "lq_path": str(lq_path),
            "gt_path": str(gt_path),
        }
