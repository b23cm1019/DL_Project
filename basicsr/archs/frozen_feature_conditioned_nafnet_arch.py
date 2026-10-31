"""NAFNet decoder conditioned by frozen shared-feature severity predictions."""
import torch.nn.functional as F
from torch import nn

from basicsr.archs.nafnet_arch import NAFNetBaseline
from basicsr.utils.registry import ARCH_REGISTRY


class _SeverityFiLM(nn.Module):
    def __init__(self, condition_dim, channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(condition_dim, channels * 2)
        # The initial network is exactly the unconditioned NAFNet checkpoint.
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, feature, condition):
        scale, shift = self.to_scale_shift(condition).chunk(2, dim=1)
        return feature * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


@ARCH_REGISTRY.register()
class FrozenFeatureConditionedNAFNet(NAFNetBaseline):
    """Condition only after the frozen extractor feature tap.

    The encoder/middle weights remain compatible with a standard NAFNet
    checkpoint.  The owning Stage-3 model freezes that analysis trunk and the
    extractor, while this class trains its condition projector, FiLM adapters,
    and decoder.  Conditioning is intentionally absent from encoder blocks so
    it cannot alter the feature distribution seen by the frozen extractor.
    """

    def __init__(self, condition_dim=16, adapter_dim=128, *args, **kwargs):
        super().__init__(*args, prompt_dim=0, **kwargs)
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, adapter_dim), nn.GELU(), nn.Linear(adapter_dim, adapter_dim)
        )
        middle_channels = self.intro.out_channels * (2 ** len(self.encoders))
        decoder_channels = [
            self.intro.out_channels * (2 ** level)
            for level in reversed(range(len(self.encoders)))
        ]
        self.middle_film = _SeverityFiLM(adapter_dim, middle_channels)
        self.decoder_film = nn.ModuleList([
            _SeverityFiLM(adapter_dim, channels) for channels in decoder_channels
        ])

    def forward(self, image, condition=None, return_degradation_features=False):
        height, width = image.shape[-2:]
        multiple = 2 ** len(self.encoders)
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.intro(image)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)
        # This is the pre-adapter analysis tap shared with the frozen
        # extractor.  It deliberately excludes all Stage-3 conditioning.
        if return_degradation_features:
            return skips, self.middle_blks(x)
        if condition is None:
            raise ValueError("condition is required for restoration output")
        projected = self.condition_projector(condition)
        x = self.middle_film(self.middle_blks(x), projected)
        for index, (up, skip, film) in enumerate(zip(self.ups, reversed(skips), self.decoder_film)):
            x = up(x) + skip
            x = film(getattr(self, f"decoder{index}")(x), projected)
        return (self.ending(x) + image)[..., :height, :width]
