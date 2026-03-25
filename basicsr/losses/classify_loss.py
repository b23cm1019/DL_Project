from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v!r}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.0)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


@LOSS_REGISTRY.register()
class MultiLabelBCELoss(nn.Module):
    """Multi-Label BCE Loss for compound degradation classification.

    Converts single integer dataset_idx to multi-hot vectors using a label_map,
    then applies BCEWithLogitsLoss. This enables the encoder to learn that an
    image can have multiple simultaneous degradations (e.g., haze AND rain).

    The label_map maps each dataset index to a list of primitive degradation
    indices that are present. For CDD-11 with 4 primitives [low, haze, rain, snow]:
        0: low       -> [1, 0, 0, 0]
        1: haze      -> [0, 1, 0, 0]
        2: rain      -> [0, 0, 1, 0]
        3: snow      -> [0, 0, 0, 1]
        4: low+haze  -> [1, 1, 0, 0]
        ...etc.
    """

    def __init__(
        self,
        label_map: list = None,
        num_primitives: int = 4,
        reduction: str = "mean",
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.num_primitives = num_primitives
        self.reduction = reduction

        # Default CDD-11 label map: 11 compound categories -> 4 primitives
        # Primitives: [low-light, haze, rain, snow]
        if label_map is None:
            label_map = [
                [1, 0, 0, 0],  # 0: low
                [0, 1, 0, 0],  # 1: haze
                [0, 0, 1, 0],  # 2: rain
                [0, 0, 0, 1],  # 3: snow
                [1, 1, 0, 0],  # 4: low+haze
                [1, 0, 1, 0],  # 5: low+rain
                [1, 0, 0, 1],  # 6: low+snow
                [0, 1, 1, 0],  # 7: haze+rain
                [0, 1, 0, 1],  # 8: haze+snow
                [1, 1, 1, 0],  # 9: low+haze+rain
                [1, 1, 0, 1],  # 10: low+haze+snow
            ]

        self.register_buffer(
            "label_map_tensor",
            torch.tensor(label_map, dtype=torch.float32),
        )

        pw = torch.ones(num_primitives) * pos_weight
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=pw
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, num_primitives) - raw logits from classifier
            y: (batch_size,) - integer dataset indices
        Returns:
            BCE loss between sigmoid(x) and multi-hot targets
        """
        # Convert integer labels to multi-hot vectors
        multi_hot = self.label_map_tensor[y.long()]  # (batch_size, num_primitives)
        return self.bce_loss(x, multi_hot)
