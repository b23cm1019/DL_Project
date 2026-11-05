"""Dataset variants for the ablations.

AblationConditionedPairedDataset (Stage 3) = SeverityConditionedPairedDataset plus
  condition_transform  none | type_only (S1) | oracle_type, applied to the cached logits
  exclude_settings     CDD-11 settings removed from training (X3 held-out combination)

AblationSeverityDCPTPairedDataset (Stage 1/2) = SeverityDCPTPairedDataset plus
  exclude_settings     CDD-11 settings removed (strict X3: Stage 2 never sees them either)
"""
import torch

from basicsr.data.severity_conditioned_paired_dataset import SeverityConditionedPairedDataset
from basicsr.data.severity_dcpt_dataset import SeverityDCPTPairedDataset
from basicsr.utils.registry import DATASET_REGISTRY

from ablations.ablib.conditions import apply_transform


def _record_setting(record):
    return record.get("setting") or record.get("composition") or "_".join(record.get("active_degradations", []))


def _filter(records, exclude, manifest):
    exclude = set(exclude or [])
    if not exclude:
        return records
    kept = [r for r in records if _record_setting(r) not in exclude]
    if not kept:
        raise ValueError(f"exclude_settings {sorted(exclude)} removed every record of {manifest}")
    return kept


@DATASET_REGISTRY.register()
class AblationConditionedPairedDataset(SeverityConditionedPairedDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.records = _filter(self.records, opt.get("exclude_settings"), opt["manifest"])
        self.condition_transform = opt.get("condition_transform", "none")
        apply_transform(self.condition_transform, torch.zeros(4, 4), "low")  # fail early on a bad name

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if self.condition_transform not in (None, "", "none"):
            item["logits"] = apply_transform(self.condition_transform, item["logits"], item["setting"])
        return item


@DATASET_REGISTRY.register()
class AblationSeverityDCPTPairedDataset(SeverityDCPTPairedDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.excluded_settings = tuple(opt.get("exclude_settings") or ())  # read by the Stage 2 sampler
        self.records = _filter(self.records, self.excluded_settings, opt["manifest"])
