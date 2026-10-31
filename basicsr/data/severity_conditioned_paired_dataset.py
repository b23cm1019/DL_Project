"""CDD-11 pairs with cached full-image severity logits (Stage 3b).

Every random training crop carries the logits of its *full* source image, which
is also what the analysis copy produces at test time, so train and test
conditions match.
"""
import json
from pathlib import Path

import torch
from torch.utils import data

from basicsr.data.severity_dcpt_dataset import _read_rgb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SeverityConditionedPairedDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.phase = opt.get("phase", "train")
        self.gt_size = int(opt.get("gt_size", 128))
        with Path(opt["manifest"]).open() as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"Empty manifest: {opt['manifest']}")
        if "logits" not in self.records[0]:
            raise ValueError("Manifest lacks cached logits; run cache_severity_full_conditions.py")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        lq, gt = _read_rgb(record["lq_path"]), _read_rgb(record["gt_path"])
        if self.phase == "train":
            gt, lq = paired_random_crop(gt, lq, self.gt_size, 1, record["lq_path"])
            gt, lq = augment([gt, lq], self.opt.get("use_hflip", True), self.opt.get("use_rot", True))
        return {
            "lq": torch.from_numpy(lq.transpose(2, 0, 1).copy()).float(),
            "gt": torch.from_numpy(gt.transpose(2, 0, 1).copy()).float(),
            "logits": torch.tensor(record["logits"], dtype=torch.float32),
            "setting": record["setting"],
            "lq_path": record["lq_path"],
        }
