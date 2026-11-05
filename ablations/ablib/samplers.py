"""Stage 2 batch sampler that accepts deliberately held-out CDD-11 settings (X3, strict).

basicsr's FixedMixedSourceBatchSampler requires all 11 CDD-11 settings in each source. This
subclass is identical except that a source may miss exactly the settings its dataset lists in
``excluded_settings`` (set by AblationSeverityDCPTPairedDataset). Any other missing setting is
still an error.
"""
from collections import defaultdict

from basicsr.data.data_sampler import FixedMixedSourceBatchSampler, _category


class HoldoutFixedMixedSourceBatchSampler(FixedMixedSourceBatchSampler):
    @staticmethod
    def _source_groups(dataset):
        if not hasattr(dataset, "datasets"):
            raise TypeError("FixedMixedSourceBatchSampler requires ConcatDataset")
        grouped = {"cdd11": defaultdict(list), "synthetic": defaultdict(list)}
        excluded = {"cdd11": set(), "synthetic": set()}
        offset = 0
        for child in dataset.datasets:
            source = getattr(child, "source", None)
            if source not in grouped:
                raise ValueError("Stage 2 children must declare source='cdd11' or source='synthetic'")
            records = getattr(child, "records", None)
            if records is None:
                raise TypeError("Stage 2 datasets must expose manifest records")
            excluded[source] |= set(getattr(child, "excluded_settings", ()))
            for index, record in enumerate(records):
                grouped[source][_category(record)].append(offset + index)
            offset += len(child)
        for source, categories in grouped.items():
            expected = 11 - len(excluded[source])
            if len(categories) != expected or set(categories) & excluded[source]:
                raise ValueError(f"{source}: expected {expected} CDD-11 settings (held out: {sorted(excluded[source])}); "
                                 f"found {sorted(categories)}")
        return {source: dict(categories) for source, categories in grouped.items()}
