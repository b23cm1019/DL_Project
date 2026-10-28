"""Restormer backbones that satisfy the severity-aware feature-tap contract.

The proposed method is backbone-agnostic at the algorithm level: a restoration
backbone must expose unconditioned encoder/latent features for the extractor
and accept conditioning only *after* that feature tap.  This module implements
that contract for the standard Restormer topology without modifying the
unconditioned Restormer parameter names or computation.
"""

import torch
import torch.nn.functional as F
from torch import nn

from basicsr.archs.restormer_arch import Restormer
from basicsr.utils.registry import ARCH_REGISTRY


class _SeverityFiLM(nn.Module):
    """Zero-initialized channel-wise FiLM preserving the base checkpoint."""

    def __init__(self, condition_dim, channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, feature, condition):
        scale, shift = self.to_scale_shift(condition).chunk(2, dim=1)
        return feature * (1 + scale[:, :, None, None]) + shift[:, :, None, None]


@ARCH_REGISTRY.register()
class FeatureTapRestormer(Restormer):
    """Standard Restormer with a stable, unconditioned feature-tap API.

    ``return_degradation_features=True`` returns the three encoder outputs and
    the latent feature.  Normal restoration is exactly the standard Restormer
    path, including its residual output.  Inputs are reflect-padded to the
    three-level down/up-sampling factor and cropped back after restoration.
    """

    _analysis_multiple = 8

    def _pad_input(self, image):
        height, width = image.shape[-2:]
        pad_h = (-height) % self._analysis_multiple
        pad_w = (-width) % self._analysis_multiple
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
        return image, height, width

    def _encode(self, image):
        out_enc_level1 = self.encoder_level1(self.patch_embed(image))
        out_enc_level2 = self.encoder_level2(self.down1_2(out_enc_level1))
        out_enc_level3 = self.encoder_level3(self.down2_3(out_enc_level2))
        latent = self.latent(self.down3_4(out_enc_level3))
        return (out_enc_level1, out_enc_level2, out_enc_level3), latent

    def _decode(self, image, encoder_features, latent):
        out_enc_level1, out_enc_level2, out_enc_level3 = encoder_features
        inp_dec_level3 = self.up4_3(latent)
        out_dec_level3 = self.decoder_level3(
            self.reduce_chan_level3(torch.cat([inp_dec_level3, out_enc_level3], 1))
        )
        inp_dec_level2 = self.up3_2(out_dec_level3)
        out_dec_level2 = self.decoder_level2(
            self.reduce_chan_level2(torch.cat([inp_dec_level2, out_enc_level2], 1))
        )
        inp_dec_level1 = self.up2_1(out_dec_level2)
        out_dec_level1 = self.decoder_level1(torch.cat([inp_dec_level1, out_enc_level1], 1))
        out_dec_level1 = self.refinement(out_dec_level1)
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(out_enc_level1)
            return self.output(out_dec_level1)
        if self.scale != 1:
            raise NotImplementedError("Severity-aware Restormer currently supports scale=1 restoration only")
        return self.output(out_dec_level1) + image

    def forward(self, image, return_degradation_features=False):
        image, height, width = self._pad_input(image)
        encoder_features, latent = self._encode(image)
        if return_degradation_features:
            return list(encoder_features), latent
        return self._decode(image, encoder_features, latent)[..., :height, :width]


@ARCH_REGISTRY.register()
class FrozenFeatureConditionedRestormer(FeatureTapRestormer):
    """Restormer decoder conditioned after an immutable extractor feature tap.

    Stage 3 freezes the feature-tap encoder/latent path and the ordinal
    extractor.  Only the projector, zero-initialized FiLM modules, decoder,
    refinement blocks, upsampling and output layer are trainable.  Because all
    adapters start at zero, loading a Stage-2 ``FeatureTapRestormer`` produces
    exactly its unconditioned restoration output before Stage-3 training.
    """

    def __init__(self, condition_dim=16, adapter_dim=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.patch_embed.proj.out_channels
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, adapter_dim), nn.GELU(), nn.Linear(adapter_dim, adapter_dim)
        )
        self.latent_film = _SeverityFiLM(adapter_dim, dim * 8)
        self.decoder_level3_film = _SeverityFiLM(adapter_dim, dim * 4)
        self.decoder_level2_film = _SeverityFiLM(adapter_dim, dim * 2)
        self.decoder_level1_film = _SeverityFiLM(adapter_dim, dim * 2)
        self.refinement_film = _SeverityFiLM(adapter_dim, dim * 2)

    def _decode_conditioned(self, image, encoder_features, latent, condition):
        out_enc_level1, out_enc_level2, out_enc_level3 = encoder_features
        projected = self.condition_projector(condition)
        latent = self.latent_film(latent, projected)
        inp_dec_level3 = self.up4_3(latent)
        out_dec_level3 = self.decoder_level3(
            self.reduce_chan_level3(torch.cat([inp_dec_level3, out_enc_level3], 1))
        )
        out_dec_level3 = self.decoder_level3_film(out_dec_level3, projected)
        inp_dec_level2 = self.up3_2(out_dec_level3)
        out_dec_level2 = self.decoder_level2(
            self.reduce_chan_level2(torch.cat([inp_dec_level2, out_enc_level2], 1))
        )
        out_dec_level2 = self.decoder_level2_film(out_dec_level2, projected)
        inp_dec_level1 = self.up2_1(out_dec_level2)
        out_dec_level1 = self.decoder_level1(torch.cat([inp_dec_level1, out_enc_level1], 1))
        out_dec_level1 = self.decoder_level1_film(out_dec_level1, projected)
        out_dec_level1 = self.refinement_film(self.refinement(out_dec_level1), projected)
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(out_enc_level1)
            return self.output(out_dec_level1)
        if self.scale != 1:
            raise NotImplementedError("Severity-aware Restormer currently supports scale=1 restoration only")
        return self.output(out_dec_level1) + image

    def forward(self, image, condition=None, return_degradation_features=False):
        image, height, width = self._pad_input(image)
        encoder_features, latent = self._encode(image)
        if return_degradation_features:
            return list(encoder_features), latent
        if condition is None:
            raise ValueError("condition is required for conditioned Restormer restoration")
        return self._decode_conditioned(image, encoder_features, latent, condition)[..., :height, :width]
