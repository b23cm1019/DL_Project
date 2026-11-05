"""Stage 1 / Stage 2 models with a selectable severity loss (S5).

Same as SeverityAwareDCPTPretrainModel / SeverityExtractorAdaptationModel, except that
``train.severity_loss`` chooses the loss on the severity bins of present primitives:

  ordinal      CDF / EMD loss (the proposed method, default)
  categorical  plain cross-entropy over bins 1..3 (no order between bins)
  regression   squared error of the expected bin (use with the extractor's ``head: scalar``)

Everything else (identity loss, type loss, optimizers, validation) is unchanged.
"""
import torch
import torch.nn.functional as F

from basicsr.models.severity_aware_dcpt_model import (
    SeverityAwareDCPTPretrainModel,
    SeverityExtractorAdaptationModel,
    _SeverityDCPTBase,
)
from basicsr.utils.registry import MODEL_REGISTRY

_LOSSES = ("ordinal", "categorical", "regression")


class _SelectableSeverityLoss:
    def _per_type_ordinal_severity_loss(self, logits, bins, has_severity):
        kind = self.opt.get("train", {}).get("severity_loss", "ordinal")
        if kind not in _LOSSES:
            raise ValueError(f"train.severity_loss must be one of {_LOSSES}")
        if kind == "ordinal":
            return _SeverityDCPTBase._per_type_ordinal_severity_loss(logits, bins, has_severity)
        losses = []
        for type_index in range(logits.shape[1]):
            active = has_severity & (bins[:, type_index] > 0)
            if not active.any():
                continue
            conditional = logits[active, type_index, 1:]
            target = bins[active, type_index] - 1
            if kind == "categorical":
                losses.append(F.cross_entropy(conditional, target))
            else:
                levels = conditional.shape[-1]
                positions = torch.arange(levels, device=logits.device, dtype=conditional.dtype)
                expected = (conditional.softmax(-1) * positions).sum(-1)
                losses.append(((expected - target.to(expected.dtype)) / max(levels - 1, 1)).square().mean())
        return torch.stack(losses).mean() if losses else logits.new_zeros(())


@MODEL_REGISTRY.register()
class AblationSeverityPretrainModel(_SelectableSeverityLoss, SeverityAwareDCPTPretrainModel):
    """Stage 1 with a selectable severity loss."""


@MODEL_REGISTRY.register()
class AblationSeverityAdaptationModel(_SelectableSeverityLoss, SeverityExtractorAdaptationModel):
    """Stage 2 with a selectable severity loss (used for the synthetic replay term)."""
