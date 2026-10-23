"""Shared-trunk binary degradation-presence extractor for Stage 0a."""
import torch
from torch import nn

from basicsr.archs.ordinal_degradation_extractor_arch import _ResidualBlock
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class Stage0TypeExtractor(nn.Module):
    """Return one presence logit for each primitive degradation."""

    def __init__(self, num_types=4, widths=(64, 128, 256, 384)):
        super().__init__()
        layers = [nn.Conv2d(3, widths[0], 5, 2, 2, bias=False),
                  nn.GroupNorm(16, widths[0]), nn.GELU()]
        previous = widths[0]
        for width in widths:
            layers.extend([_ResidualBlock(previous, width, 1), _ResidualBlock(width, width, 2)])
            previous = width
        self.trunk = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(widths[-1], num_types)

    def forward(self, image):
        return self.head(self.trunk(image))
