"""NAFNet with extractor-probability conditioning and FiLM injection."""
import torch
from torch import nn

from basicsr.archs.nafnet_arch import NAFNetBaseline
from basicsr.utils.registry import ARCH_REGISTRY


class FiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, feature, cond):
        scale, shift = self.proj(cond).chunk(2, dim=1)
        return feature * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


@ARCH_REGISTRY.register()
class FiLMConditionedNAFNet(NAFNetBaseline):
    def __init__(self, num_types=4, num_bins=4, cond_dim=128, *args, **kwargs):
        super().__init__(*args, prompt_dim=0, **kwargs)
        self.cond_embed = nn.Sequential(nn.Linear(num_types * num_bins, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim))
        channels = [self.intro.out_channels * (2 ** index) for index in range(len(self.encoders))]
        self.encoder_film = nn.ModuleList([FiLM(cond_dim, channel) for channel in channels])
        self.middle_film = FiLM(cond_dim, channels[-1] * 2)
        self.decoder_film = nn.ModuleList([FiLM(cond_dim, channel) for channel in reversed(channels)])

    def forward(self, inp, condition):
        cond = self.cond_embed(condition.flatten(1))
        x, skips = self.intro(inp), []
        for encoder, down, film in zip(self.encoders, self.downs, self.encoder_film):
            x = film(encoder(x), cond)
            skips.append(x)
            x = down(x)
        x = self.middle_film(self.middle_blks(x), cond)
        for index, (up, skip, film) in enumerate(zip(self.ups, reversed(skips), self.decoder_film)):
            x = up(x) + skip
            x = film(getattr(self, f"decoder{index}")(x), cond)
        return self.ending(x) + inp
