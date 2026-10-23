"""Small shared-trunk extractor with independent ordinal heads."""
import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
        )
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x):
        return torch.nn.functional.gelu(self.body(x) + self.skip(x))


@ARCH_REGISTRY.register()
class OrdinalDegradationExtractor(nn.Module):
    """Direct image extractor; output shape is [batch, type, ordinal_bin]."""
    def __init__(self, num_types=4, num_bins=4, widths=(64, 128, 256, 384)):
        super().__init__()
        self.num_types, self.num_bins = num_types, num_bins
        layers = [nn.Conv2d(3, widths[0], 5, 2, 2, bias=False), nn.GroupNorm(16, widths[0]), nn.GELU()]
        previous = widths[0]
        for width in widths:
            layers.extend([_ResidualBlock(previous, width, 1), _ResidualBlock(width, width, 2)])
            previous = width
        self.trunk = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.heads = nn.ModuleList([nn.Linear(widths[-1], num_bins) for _ in range(num_types)])

    def forward(self, image):
        features = self.trunk(image)
        return torch.stack([head(features) for head in self.heads], dim=1)
