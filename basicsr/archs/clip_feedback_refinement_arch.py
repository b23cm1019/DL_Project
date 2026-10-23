"""Frozen Row-C NAFNet with a small feedback-conditioned FiLM adapter."""
from torch import nn
import torch.nn.functional as F

from basicsr.archs.nafnet_arch import NAFNetBaseline
from basicsr.utils.registry import ARCH_REGISTRY


class FeedbackFiLM(nn.Module):
    def __init__(self, condition_dim, channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(condition_dim, channels * 2)
        # Exact naïve-feedback behaviour at initialization: F(y1).
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x, condition):
        scale, shift = self.to_scale_shift(condition).chunk(2, dim=1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


@ARCH_REGISTRY.register()
class CLIPFeedbackRefinementNAFNet(NAFNetBaseline):
    """A NAFNet whose original weights are frozen by the owning model.

    The adapter is intentionally additive to a Row-C-compatible NAFNet state
    dictionary. Loading the Row-C checkpoint with ``strict=False`` loads the
    original backbone and leaves only conditioner/FiLM parameters new.
    """
    def __init__(self, feedback_dim=16, adapter_dim=128, *args, **kwargs):
        super().__init__(*args, prompt_dim=0, **kwargs)
        self.feedback_projector = nn.Sequential(
            nn.Linear(feedback_dim, adapter_dim), nn.GELU(), nn.Linear(adapter_dim, adapter_dim)
        )
        channels = [self.intro.out_channels * (2**level) for level in range(len(self.encoders))]
        self.encoder_film = nn.ModuleList([FeedbackFiLM(adapter_dim, channel) for channel in channels])
        self.middle_film = FeedbackFiLM(adapter_dim, channels[-1] * 2)
        self.decoder_film = nn.ModuleList([FeedbackFiLM(adapter_dim, channel) for channel in reversed(channels)])

    def forward(self, image, feedback):
        height, width = image.shape[-2:]
        multiple = 2 ** len(self.encoders)
        pad_h, pad_w = (-height) % multiple, (-width) % multiple
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        condition = self.feedback_projector(feedback)
        x, skips = self.intro(image), []
        for encoder, down, film in zip(self.encoders, self.downs, self.encoder_film):
            x = film(encoder(x), condition)
            skips.append(x)
            x = down(x)
        x = self.middle_film(self.middle_blks(x), condition)
        for index, (up, skip, film) in enumerate(zip(self.ups, reversed(skips), self.decoder_film)):
            x = up(x) + skip
            x = film(getattr(self, f"decoder{index}")(x), condition)
        return (self.ending(x) + image)[..., :height, :width]
