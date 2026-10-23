import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class OrdinalFocalCrossEntropyLoss(nn.Module):
    """Independent focal CE for each degradation type's ordinal-bin head."""
    def __init__(self, gamma=2.0, class_weights=None, loss_weight=1.0):
        super().__init__()
        self.gamma, self.loss_weight = gamma, loss_weight
        self.register_buffer("class_weights", None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32))

    def forward(self, logits, targets):
        if logits.ndim != 3 or targets.shape != logits.shape[:2]:
            raise ValueError("Expected logits [B,T,K] and targets [B,T]")
        losses = []
        for type_index in range(logits.shape[1]):
            weights = self.class_weights
            ce = F.cross_entropy(logits[:, type_index], targets[:, type_index], weight=weights, reduction="none")
            losses.append((((1 - torch.exp(-ce)) ** self.gamma) * ce).mean())
        return self.loss_weight * torch.stack(losses).mean()
