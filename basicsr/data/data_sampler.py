import math
import random
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler


class EnlargedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    Modified from torch.utils.data.distributed.DistributedSampler
    Support enlarging the dataset for iteration-based training, for saving
    time when restart the dataloader after each epoch

    Args:
        dataset (torch.utils.data.Dataset): Dataset used for sampling.
        num_replicas (int | None): Number of processes participating in
            the training. It is usually the world_size.
        rank (int | None): Rank of the current process within num_replicas.
        ratio (int): Enlarging ratio. Default: 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()

        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def _category(record):
    """Stable CDD-11 category key from a synthetic or CDD manifest record."""
    active = record.get("active_degradations")
    if active is None:
        active = [part for part in record.get("setting", "").split("_") if part]
    return "_".join(active)


class CompositionBalancedSampler(Sampler):
    """Distributed sampler that cycles evenly through manifest compositions.

    The ordinary ``EnlargedSampler`` randomly permutes records.  That is only
    approximately balanced when category counts happen to agree.  This sampler
    is for Stage 1, where each pass explicitly interleaves all CDD-11
    compositions while retaining random ordering inside every composition.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) * ratio / num_replicas)
        self.total_size = self.num_samples * num_replicas
        self.groups = self._groups(dataset)
        if len(self.groups) < 2:
            raise ValueError("CompositionBalancedSampler requires manifest-backed composition records")

    @staticmethod
    def _groups(dataset):
        groups = defaultdict(list)
        if hasattr(dataset, "datasets"):
            offset = 0
            for child in dataset.datasets:
                records = getattr(child, "records", None)
                if records is None:
                    raise TypeError("Every child dataset must expose manifest records")
                for index, record in enumerate(records):
                    groups[_category(record)].append(offset + index)
                offset += len(child)
        else:
            records = getattr(dataset, "records", None)
            if records is None:
                raise TypeError("Dataset must expose manifest records")
            for index, record in enumerate(records):
                groups[_category(record)].append(index)
        return {name: values for name, values in groups.items() if values}

    def __iter__(self):
        rng = random.Random(self.epoch)
        names = sorted(self.groups)
        pools = {name: list(indices) for name, indices in self.groups.items()}
        positions = {name: 0 for name in names}
        for values in pools.values():
            rng.shuffle(values)
        indices = []
        while len(indices) < self.total_size:
            cycle = list(names)
            rng.shuffle(cycle)
            for name in cycle:
                if positions[name] == len(pools[name]):
                    rng.shuffle(pools[name])
                    positions[name] = 0
                indices.append(pools[name][positions[name]])
                positions[name] += 1
                if len(indices) == self.total_size:
                    break
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class FixedMixedSourceBatchSampler(Sampler):
    """Yield fixed CDD/synthetic batches for Stage 2 replay.

    Every yielded batch contains exactly ``cdd_per_batch`` CDD examples and
    ``synthetic_per_batch`` synthetic examples.  Within either source, records
    are cycled uniformly over their CDD-11 composition categories.  Sampling
    with replacement across an epoch is deliberate: the CDD set is much
    smaller than the external synthetic corpus, but both must contribute equal
    sample counts to every adaptation update.
    """

    def __init__(self, dataset, num_replicas, rank, batch_size, cdd_per_batch, synthetic_per_batch):
        if batch_size != cdd_per_batch + synthetic_per_batch:
            raise ValueError("batch_size must equal cdd_per_batch + synthetic_per_batch")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.cdd_per_batch = cdd_per_batch
        self.synthetic_per_batch = synthetic_per_batch
        self.epoch = 0
        # ``num_batches`` is local to one DDP rank.  A global optimizer step
        # therefore consumes ``batch_size * num_replicas`` records while each
        # rank still receives the requested fixed source mix.
        self.num_batches = math.ceil(len(dataset) / (batch_size * num_replicas))
        self.groups = self._source_groups(dataset)

    @staticmethod
    def _source_groups(dataset):
        if not hasattr(dataset, "datasets"):
            raise TypeError("FixedMixedSourceBatchSampler requires ConcatDataset")
        grouped = {"cdd11": defaultdict(list), "synthetic": defaultdict(list)}
        offset = 0
        for child in dataset.datasets:
            source = getattr(child, "source", None)
            if source not in grouped:
                raise ValueError("Stage 2 children must declare source='cdd11' or source='synthetic'")
            records = getattr(child, "records", None)
            if records is None:
                raise TypeError("Stage 2 datasets must expose manifest records")
            for index, record in enumerate(records):
                grouped[source][_category(record)].append(offset + index)
            offset += len(child)
        for source, categories in grouped.items():
            if len(categories) != 11:
                raise ValueError(f"{source} must contain all 11 degraded CDD-11 settings; found {sorted(categories)}")
        return {source: dict(categories) for source, categories in grouped.items()}

    @staticmethod
    def _draw(groups, count, rng, pools, positions):
        names = sorted(groups)
        output = []
        while len(output) < count:
            cycle = list(names)
            rng.shuffle(cycle)
            for name in cycle:
                key = (id(groups), name)
                if positions[key] == len(pools[key]):
                    rng.shuffle(pools[key])
                    positions[key] = 0
                output.append(pools[key][positions[key]])
                positions[key] += 1
                if len(output) == count:
                    break
        return output

    def __iter__(self):
        rng = random.Random(self.epoch)
        pools, positions = {}, {}
        for groups in self.groups.values():
            for name, values in groups.items():
                key = (id(groups), name)
                pools[key] = list(values)
                rng.shuffle(pools[key])
                positions[key] = 0
        # Build a shared deterministic global stream, then shard whole batches
        # by rank.  Sharding individual samples would destroy the required
        # per-rank CDD/synthetic ratio; sharding batches preserves it and avoids
        # duplicate local batches across ranks.
        for global_batch_index in range(self.num_batches * self.num_replicas):
            batch = self._draw(self.groups["cdd11"], self.cdd_per_batch, rng, pools, positions)
            batch += self._draw(self.groups["synthetic"], self.synthetic_per_batch, rng, pools, positions)
            rng.shuffle(batch)
            if global_batch_index % self.num_replicas == self.rank:
                yield batch

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch
