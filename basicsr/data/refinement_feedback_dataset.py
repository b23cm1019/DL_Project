"""Cached pass-1 samples and feedback vectors for one-step refinement."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils import data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.registry import DATASET_REGISTRY


def _image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


@DATASET_REGISTRY.register()
class RefinementFeedbackDataset(data.Dataset):
    """Manifest fields: pass1_path, gt_path, clip_path, oracle_condition."""
    def __init__(self, opt):
        self.opt = opt
        with open(opt["manifest"]) as handle:
            self.items = [json.loads(line) for line in handle if line.strip()]
        if not self.items:
            raise ValueError(f"Empty refinement manifest: {opt['manifest']}")
        self.gt_size = opt.get("gt_size", 128)
        self.mode = opt.get("feedback_mode", "clip")
        self.shuffle_seed = opt.get("shuffle_seed", 0)
        self.permutation = np.random.default_rng(self.shuffle_seed).permutation(len(self.items))

    def _feedback(self, index, item):
        if self.mode == "oracle":
            return np.asarray(item["oracle_condition"], dtype=np.float32)
        source = self.items[self.permutation[index]] if self.mode == "shuffled_clip" else item
        return np.load(source["clip_path"]).astype(np.float32)

    def __getitem__(self, index):
        item = self.items[index]
        pass1, gt = _image(item["pass1_path"]), _image(item["gt_path"])
        if self.opt["phase"] == "train":
            gt, pass1 = paired_random_crop(gt, pass1, self.gt_size, 1, item["pass1_path"])
            pass1, gt = augment([pass1, gt], self.opt.get("use_hflip", True), self.opt.get("use_rot", True))
        pass1 = torch.from_numpy(pass1.transpose(2, 0, 1)).float().contiguous()
        gt = torch.from_numpy(gt.transpose(2, 0, 1)).float().contiguous()
        return {"lq": pass1, "gt": gt, "feedback": torch.from_numpy(self._feedback(index, item)), "lq_path": item["pass1_path"]}

    def __len__(self):
        return len(self.items)
