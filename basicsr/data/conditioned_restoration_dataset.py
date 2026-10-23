"""Original-image restoration dataset with a cached conditioning vector."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.registry import DATASET_REGISTRY


def _read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


@DATASET_REGISTRY.register()
class ConditionedRestorationDataset(data.Dataset):
    """Manifest fields: ``lq_path``, ``gt_path`` and ``condition_path``.

    The condition is computed once by a frozen teacher before training.  The
    restoration input remains the original degraded image (not Row-C pass-1).
    ``condition_mode`` supports direct, clip, zero, shuffled, and random
    controls.  Shuffling/randomization is deterministic per dataset instance.
    """

    def __init__(self, opt):
        self.opt = opt
        with open(opt["manifest"]) as handle:
            self.items = [json.loads(line) for line in handle if line.strip()]
        if not self.items:
            raise ValueError(f"Empty conditioned manifest: {opt['manifest']}")
        self.gt_size = opt.get("gt_size", 128)
        self.mode = opt.get("condition_mode", "clip")
        self.condition_dim = int(opt["condition_dim"])
        self.seed = int(opt.get("condition_seed", 2026))
        self.permutation = np.random.default_rng(self.seed).permutation(len(self.items))
        self._stats = self._load_stats()

    def _load_stats(self):
        values = []
        for item in self.items:
            condition = np.load(item["condition_path"]).astype(np.float32)
            if condition.shape != (self.condition_dim,):
                raise ValueError(
                    f"Expected condition shape {(self.condition_dim,)}, "
                    f"got {condition.shape} for {item['condition_path']}"
                )
            values.append(condition)
        matrix = np.stack(values)
        return matrix.mean(0), matrix.std(0).clip(min=1e-4)

    def _condition(self, index, item):
        if self.mode == "zero":
            return np.zeros(self.condition_dim, dtype=np.float32)
        if self.mode == "shuffled":
            source = self.items[self.permutation[index]]
            return np.load(source["condition_path"]).astype(np.float32)
        if self.mode == "random":
            rng = np.random.default_rng(self.seed + index)
            return rng.normal(self._stats[0], self._stats[1]).astype(np.float32)
        return np.load(item["condition_path"]).astype(np.float32)

    def __getitem__(self, index):
        item = self.items[index]
        lq, gt = _read_image(item["lq_path"]), _read_image(item["gt_path"])
        if self.opt["phase"] == "train":
            gt, lq = paired_random_crop(gt, lq, self.gt_size, 1, item["lq_path"])
            lq, gt = augment(
                [lq, gt], self.opt.get("use_hflip", True), self.opt.get("use_rot", True)
            )
        lq = torch.from_numpy(lq.transpose(2, 0, 1)).float().contiguous()
        gt = torch.from_numpy(gt.transpose(2, 0, 1)).float().contiguous()
        condition = torch.from_numpy(self._condition(index, item)).float()
        return {
            "lq": lq,
            "gt": gt,
            "condition": condition,
            "lq_path": item["lq_path"],
            "gt_path": item["gt_path"],
        }

    def __len__(self):
        return len(self.items)
