"""Ablation variants of the Stage 3 restorer and of the severity extractor.

AblationFullNAFNet extends SeverityConditionedFullNAFNet (the Plan B restorer). With default
arguments it is the same network as Plan B. Options:

  film_sites  where the condition is injected: any of middle / decoder / skip      (F1, F2)
  injection   film   : scale and shift (Plan B)
              shift  : shift only. Concatenating the condition to a feature map and
                       applying a 1x1 conv equals feature + W*e, so this is the
                       concatenation baseline                                    (F3)
              prompt : PromptIR-style prompt block: learned prompt components mixed
                       by weights computed from the severity embedding, resized,
                       3x3 conv, concatenated with the feature, zero-init 1x1 fuse (F3)
  film_init   zero (Plan B: starts as the Stage 2 network) | random (PyTorch default) (F4)
  fixed_null  True: the null condition is a fixed uniform distribution, not learned  (K5)
  analysis    frozen : condition from the cached frozen analysis copy (Plan B)
              shared : condition from THIS network's own trainable encoder features,
                       read by a frozen copy of the Stage 2 extractor              (S4)

AblationSeverityExtractor extends FeatureSeverityExtractor with:
  use_levels  which feature levels feed the extractor (indices into
              [enc1, enc2, enc3, enc4, middle]; default all)                      (S10)
  head        ordinal / categorical: same [B,T,K] logits (the loss decides)
              scalar : one severity number per primitive, mapped back to the same
                       4-bin interface so Stage 3 is unchanged                      (S5)
"""
import torch
import torch.nn.functional as F
from torch import nn

from basicsr.archs.feature_severity_extractor_arch import FeatureSeverityExtractor
from basicsr.archs.frozen_feature_conditioned_nafnet_arch import _SeverityFiLM
from basicsr.archs.severity_conditioned_full_nafnet_arch import SeverityConditionedFullNAFNet
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.severity_full_analysis import EXTRACTOR_ARGS, load_state

_SITES = ("middle", "decoder", "skip")
_INJECTIONS = ("film", "shift", "prompt")


class ShiftCondition(nn.Module):
    """feature + W e  (== concatenating e to the feature map, then a 1x1 conv). Zero-init."""

    def __init__(self, condition_dim, channels):
        super().__init__()
        self.to_shift = nn.Linear(condition_dim, channels)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, feature, condition):
        return feature + self.to_shift(condition)[:, :, None, None]


class PromptCondition(nn.Module):
    """PromptIR-style prompt block driven by the severity embedding. Zero-init output."""

    def __init__(self, condition_dim, channels, prompt_len=5, prompt_size=16, prompt_channels=128):
        super().__init__()
        width = min(channels, prompt_channels)
        self.prompt = nn.Parameter(torch.rand(prompt_len, width, prompt_size, prompt_size))
        self.to_weights = nn.Linear(condition_dim, prompt_len)
        self.prompt_conv = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.fuse = nn.Conv2d(channels + width, channels, 1)
        nn.init.zeros_(self.fuse.weight)
        nn.init.zeros_(self.fuse.bias)

    def forward(self, feature, condition):
        weights = self.to_weights(condition).softmax(-1)
        prompt = torch.einsum("bn,nchw->bchw", weights, self.prompt)
        prompt = F.interpolate(prompt, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        prompt = self.prompt_conv(prompt)
        return feature + self.fuse(torch.cat([feature, prompt], 1))


@ARCH_REGISTRY.register()
class AblationFullNAFNet(SeverityConditionedFullNAFNet):
    def __init__(self, *args, film_sites=_SITES, injection="film", film_init="zero", fixed_null=False,
                 analysis="frozen", shared_extractor=None, shared_extractor_path=None, shared_detach=True,
                 prompt_len=5, prompt_size=16, prompt_channels=128, **kwargs):
        # The parent runs one dummy forward (TLC conversion) inside __init__; until
        # _ablation_ready is set, forward() falls back to the parent's forward.
        super().__init__(*args, **kwargs)
        sites = set(film_sites)
        if not sites <= set(_SITES):
            raise ValueError(f"film_sites must be a subset of {_SITES}, got {film_sites}")
        if injection not in _INJECTIONS:
            raise ValueError(f"injection must be one of {_INJECTIONS}")
        if film_init not in ("zero", "random"):
            raise ValueError("film_init must be 'zero' or 'random'")
        if analysis not in ("frozen", "shared"):
            raise ValueError("analysis must be 'frozen' or 'shared'")
        adapter_dim = self.condition_projector[-1].out_features
        middle_channels = self.middle_film.to_scale_shift.out_features // 2
        level_channels = [film.to_scale_shift.out_features // 2 for film in self.decoder_film]

        def make(channels):
            if injection == "shift":
                return ShiftCondition(adapter_dim, channels)
            if injection == "prompt":
                return PromptCondition(adapter_dim, channels, prompt_len, prompt_size, prompt_channels)
            module = _SeverityFiLM(adapter_dim, channels)
            if film_init == "random":
                module.to_scale_shift.reset_parameters()
            return module

        # Removed sites are set to None so DDP never sees unused parameters.
        self.middle_film = make(middle_channels) if "middle" in sites else None
        self.decoder_film = nn.ModuleList([make(c) for c in level_channels]) if "decoder" in sites else None
        self.skip_film = nn.ModuleList([make(c) for c in level_channels]) if "skip" in sites else None

        if fixed_null:
            del self.null_logits
            self.register_buffer("null_logits", torch.zeros(self.num_types, self.num_bins))

        self.analysis = analysis
        self.shared_detach = bool(shared_detach)
        if analysis == "shared":
            self.shared_extractor = FeatureSeverityExtractor(**(shared_extractor or EXTRACTOR_ARGS))
            if shared_extractor_path:
                self.shared_extractor.load_state_dict(load_state(shared_extractor_path), strict=True)
        self._ablation_ready = True

    @staticmethod
    def _inject(module, feature, embedding):
        if module is None or embedding is None:
            return feature
        return module(feature, embedding)

    def forward(self, image, logits=None, condition_keep=None, return_degradation_features=False,
                return_condition=False):
        if not getattr(self, "_ablation_ready", False):
            return super().forward(image, logits, condition_keep, return_degradation_features, return_condition)
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
        middle = self.middle_blks(x)
        if return_degradation_features:
            return skips, middle

        if self.analysis == "shared":
            features = [s.detach() for s in skips] if self.shared_detach else skips
            logits = self.shared_extractor(features, middle.detach() if self.shared_detach else middle)
        embedding, info = self._condition(logits, image.shape[0], image.device, condition_keep)
        x = self._inject(self.middle_film, middle, embedding)
        for index, (up, skip) in enumerate(zip(self.ups, reversed(skips))):
            skip_module = self.skip_film[index] if self.skip_film is not None else None
            x = up(x) + self._inject(skip_module, skip, embedding)
            x = getattr(self, f"decoder{index}")(x)
            decoder_module = self.decoder_film[index] if self.decoder_film is not None else None
            x = self._inject(decoder_module, x, embedding)
        output = (self.ending(x) + image)[..., :height, :width]
        if return_condition:
            return output, info
        return output


def scalar_to_bins(raw, sigma=0.5):
    """[B,T,K] raw head output -> 4-bin logits carrying one severity number per primitive.

    raw[..., 0] is the absence logit; raw[..., 1] gives mu = 1 + (K-2) * sigmoid(r) in [1, K-1].
    Present-bin logits are a fixed Gaussian bump around mu, normalized so that
    softmax(out)[..., 0] == sigmoid(raw[..., 0]). The other raw channels are unused.
    """
    num_bins = raw.shape[-1]
    centers = torch.arange(1, num_bins, device=raw.device, dtype=raw.dtype)
    mu = 1.0 + (num_bins - 2) * torch.sigmoid(raw[..., 1])
    present = (-(centers - mu[..., None]).square() / (2 * sigma ** 2)).log_softmax(-1)
    return torch.cat([raw[..., :1], present], -1)


@ARCH_REGISTRY.register()
class AblationSeverityExtractor(FeatureSeverityExtractor):
    def __init__(self, feature_dims=(64, 128, 256, 512), middle_dim=1024, hidden_dim=256, num_types=4,
                 num_bins=4, dropout=0.0, use_levels=None, head="ordinal", scalar_sigma=0.5):
        all_dims = list(feature_dims) + [middle_dim]
        levels = list(range(len(all_dims))) if use_levels is None else [int(i) for i in use_levels]
        if not levels or any(i < 0 or i >= len(all_dims) for i in levels):
            raise ValueError(f"use_levels must index {len(all_dims)} feature levels, got {use_levels}")
        if head not in ("ordinal", "categorical", "scalar"):
            raise ValueError("head must be ordinal, categorical or scalar")
        selected = [all_dims[i] for i in levels]
        super().__init__(feature_dims=selected[:-1], middle_dim=selected[-1], hidden_dim=hidden_dim,
                         num_types=num_types, num_bins=num_bins, dropout=dropout)
        self.use_levels = levels
        self.head = head
        self.scalar_sigma = float(scalar_sigma)

    def forward(self, encoder_features, middle_feature):
        features = list(encoder_features) + [middle_feature]
        selected = [features[i] for i in self.use_levels]
        logits = super().forward(selected[:-1], selected[-1])
        if self.head == "scalar":
            logits = scalar_to_bins(logits, self.scalar_sigma)
        return logits
