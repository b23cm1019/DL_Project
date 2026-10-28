"""Sampler-aware entry point for the severity-aware three-stage experiments.

The generic all-in-one trainer is shared with earlier user experiments and is
left untouched.  This entry point installs the two additional samplers needed
by the new protocol, then invokes that trainer unchanged.
"""
import math
from copy import deepcopy
from os import path as osp

import basicsr.all_in_one_train as base
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.concat_dataset import ConcatDataset
from basicsr.data.data_sampler import (
    CompositionBalancedSampler,
    EnlargedSampler,
    FixedMixedSourceBatchSampler,
)


def create_train_val_dataloader(opt, logger):
    train_sets, val_sets, enlarge_ratios = [], [], []
    for phase, dataset_opt in opt["datasets"].items():
        # All-in-one configurations historically inferred the split from the
        # key (train_1, val_1, ...).  Manifest datasets also need that split
        # inside __getitem__ to disable crop/augmentation at validation.
        dataset_opt = deepcopy(dataset_opt)
        dataset_opt.setdefault("phase", phase.split("_")[0])
        if phase.split("_")[0] == "train":
            train_set = build_dataset(dataset_opt)
            train_sets.append(train_set)
            enlarge_ratios.append(dataset_opt.get("dataset_enlarge_ratio", 1))
        elif phase.split("_")[0] == "val":
            val_sets.append(build_dataset(dataset_opt))
        else:
            raise ValueError(f"Dataset phase {phase} is not recognized.")

    dataloader_opt = opt["dataloader"]
    dataset_enlarge_ratio = dataloader_opt.get("dataset_enlarge_ratio", 1)
    train_set = ConcatDataset(train_sets, enlarge_ratios)
    sampler_type = dataloader_opt.get("sampler_type", "enlarged")
    batch_sampler = None
    if sampler_type == "enlarged":
        train_sampler = EnlargedSampler(train_set, opt["world_size"], opt["rank"], dataset_enlarge_ratio)
    elif sampler_type == "composition_balanced":
        train_sampler = CompositionBalancedSampler(train_set, opt["world_size"], opt["rank"], dataset_enlarge_ratio)
    elif sampler_type == "fixed_mixed_sources":
        train_sampler = FixedMixedSourceBatchSampler(
            train_set,
            opt["world_size"],
            opt["rank"],
            dataloader_opt["batch_size_per_gpu"],
            dataloader_opt["cdd_per_batch"],
            dataloader_opt["synthetic_per_batch"],
        )
        batch_sampler = train_sampler
    else:
        raise ValueError(f"Unsupported dataloader sampler_type: {sampler_type}")

    train_loader = build_dataloader(
        train_set,
        dataloader_opt,
        num_gpu=opt["num_gpu"],
        dist=opt["dist"],
        sampler=None if batch_sampler is not None else train_sampler,
        batch_sampler=batch_sampler,
        seed=opt["manual_seed"],
    )
    val_set = ConcatDataset(val_sets, [1] * len(val_sets))
    val_opt = opt["dataloader_val"] if "dataloader_val" in opt else dataloader_opt
    val_loader = build_dataloader(
        val_set, val_opt, num_gpu=opt["num_gpu"], dist=opt["dist"], sampler=None, seed=opt["manual_seed"]
    )
    num_iter_per_epoch = len(train_sampler) if batch_sampler is not None else math.ceil(
        len(train_set) * dataset_enlarge_ratio / (dataloader_opt["batch_size_per_gpu"] * opt["world_size"])
    )
    total_iters = int(opt["train"]["total_iter"])
    total_epochs = math.ceil(total_iters / num_iter_per_epoch)
    logger.info(
        "Training statistics:"
        f"\n\tNumber of train images: {len(train_set)}"
        f"\n\tSampler: {sampler_type}"
        f"\n\tBatch size per gpu: {dataloader_opt['batch_size_per_gpu']}"
        f"\n\tIterations per epoch: {num_iter_per_epoch}"
        f"\n\tTotal epochs: {total_epochs}; iters: {total_iters}."
    )
    logger.info(f"Number of val images/folders: {len(val_set)}")
    return train_set, train_loader, train_sampler, val_loader, total_epochs, total_iters


def create_val_dataloaders(opt, logger):
    """Build per-source validation loaders from the shared validation policy.

    ``all_in_one_train`` normally expects every individual ``val_*`` dataset
    entry to repeat loader fields.  Severity configs intentionally keep those
    fields once in ``dataloader_val``; this adapter preserves that compact,
    consistent policy for Stage 1--3 and avoids a second, incompatible loader
    construction path.
    """
    loader_opt = deepcopy(opt.get("dataloader_val", opt["dataloader"]))
    loader_opt["phase"] = "val"
    val_loaders = []
    for phase, dataset_opt in opt["datasets"].items():
        if phase.split("_")[0] != "val":
            continue
        dataset_opt = deepcopy(dataset_opt)
        dataset_opt.setdefault("phase", "val")
        val_set = build_dataset(dataset_opt)
        val_loader = build_dataloader(
            val_set,
            loader_opt,
            num_gpu=opt["num_gpu"],
            dist=opt["dist"],
            sampler=None,
            seed=opt["manual_seed"],
        )
        logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
        val_loaders.append(val_loader)
    return val_loaders


base.create_train_val_dataloader = create_train_val_dataloader
base.create_val_dataloaders = create_val_dataloaders


if __name__ == "__main__":
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    base.train_pipeline(root_path)
