"""Ordinal degradation extractor attached to restoration-backbone features."""
import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class FeatureSeverityExtractor(nn.Module):
    """Predict independent ordinal distributions from restoration features.

    The output has shape ``[B, num_types, num_bins]``.  Bin zero means that a
    primitive is absent; therefore ``1 - p(bin=0)`` is its type-presence score.
    Unlike the existing ``OrdinalDegradationExtractor``, this module has no
    image backbone: all information must come from the restoration backbone's
    unconditioned encoder/latent representation, following DCPT's
    feature-supervision principle.  The number and dimensions of feature taps
    are configured per backbone.
    """

    def __init__(
        self,
        feature_dims=(64, 128, 256, 512),
        middle_dim=1024,
        hidden_dim=256,
        num_types=4,
        num_bins=4,
        dropout=0.0,
    ):
        super().__init__()
        self.num_types = num_types
        self.num_bins = num_bins
        dims = list(feature_dims) + [middle_dim]
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            for dim in dims
        ])
        # Learned, normalized multi-level scaling: the analogous aggregation
        # principle from DCPT, applied to our encoder/middle feature taps.
        # Multiplication by the number of levels initializes every scale to 1.
        self.feature_level_logits = nn.Parameter(torch.zeros(len(dims)))
        fusion_dim = hidden_dim * len(dims)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.ordinal_head = nn.Linear(hidden_dim, num_types * num_bins)

    def forward(self, encoder_features, middle_feature):
        if len(encoder_features) != len(self.projections) - 1:
            raise ValueError(
                f"Expected {len(self.projections) - 1} encoder features, "
                f"received {len(encoder_features)}"
            )
        features = list(encoder_features) + [middle_feature]
        level_weights = len(features) * self.feature_level_logits.softmax(dim=0)
        embedding = torch.cat([
            level_weight * projection(feature)
            for level_weight, projection, feature in zip(level_weights, self.projections, features)
        ], dim=1)
        logits = self.ordinal_head(self.fusion(embedding))
        return logits.view(logits.shape[0], self.num_types, self.num_bins)
