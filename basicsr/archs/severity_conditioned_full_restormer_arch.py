"""Stage 3a/3b restorer for the Restormer backbone (backbone-agnostic proof).

Same conditioning design as ``SeverityConditionedFullNAFNet``:
  z (frozen extractor logits) -> calibrator -> p_hat -> projector -> e
  e = keep * [g * e(p_hat) + (1 - g) * e(p_null)] + (1 - keep) * e(p_null)
FiLM sites: latent ("middle_film"), decoder levels 3/2/1 + refinement
("decoder_film"), and the three encoder skips ("skip_film").  All FiLM layers
are zero-initialized, so loading a Stage-2 ``FeatureTapRestormer`` checkpoint
reproduces its unconditioned output exactly.  Module names match the NAFNet
version so ``SeverityFullFinetuneModel`` treats both identically.

Restormer has no global average pooling, so no TLC conversion is needed.
"""
import math

import torch
from torch import nn

from basicsr.archs.feature_tap_restormer_arch import FeatureTapRestormer
from basicsr.archs.frozen_feature_conditioned_nafnet_arch import _SeverityFiLM
from basicsr.archs.severity_conditioned_full_nafnet_arch import SeverityCalibrator
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class SeverityConditionedFullRestormer(FeatureTapRestormer):
    def __init__(
        self,
        *args,
        num_types=4,
        num_bins=4,
        adapter_dim=128,
        condition_mode="full",  # "full" | "constant" | "none"
        use_gate=True,
        gate_init=-6.0,
        use_calibrator=True,
        logit_temperatures=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if condition_mode not in {"full", "constant", "none"}:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")
        if self.dual_pixel_task:
            raise NotImplementedError("dual_pixel_task is not supported")
        self.condition_mode = condition_mode
        self.use_gate = use_gate
        self.use_calibrator = use_calibrator
        self.num_types, self.num_bins = num_types, num_bins
        temperatures = torch.ones(num_types) if logit_temperatures is None else torch.tensor(
            [float(t) for t in logit_temperatures])
        self.register_buffer("logit_temperatures", temperatures.view(1, num_types, 1), persistent=False)

        dim = self.patch_embed.proj.out_channels
        self.calibrator = SeverityCalibrator(num_types, num_bins)
        self.null_logits = nn.Parameter(torch.zeros(num_types, num_bins))
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.condition_projector = nn.Sequential(
            nn.Linear(num_types * num_bins, adapter_dim), nn.GELU(), nn.Linear(adapter_dim, adapter_dim)
        )
        self.middle_film = _SeverityFiLM(adapter_dim, dim * 8)  # latent
        # decoder level 3, level 2, level 1, refinement
        self.decoder_film = nn.ModuleList([_SeverityFiLM(adapter_dim, c) for c in (dim * 4, dim * 2, dim * 2, dim * 2)])
        # encoder skips level 3, level 2, level 1
        self.skip_film = nn.ModuleList([_SeverityFiLM(adapter_dim, c) for c in (dim * 4, dim * 2, dim)])

    def _condition(self, logits, batch, device, condition_keep=None):
        info = {"p_hat": None, "q": None, "gate": None}
        if self.condition_mode == "none":
            return None, info
        null_embedding = self.condition_projector(self.null_logits.softmax(-1).flatten()[None])
        null_embedding = null_embedding.to(device).expand(batch, -1)
        if self.condition_mode == "constant" or logits is None:
            return null_embedding, info
        p_hat, q = self.calibrator(logits / self.logit_temperatures, enabled=self.use_calibrator)
        embedding = self.condition_projector(p_hat.flatten(1))
        weight = torch.ones(batch, 1, device=device, dtype=embedding.dtype)
        if self.use_gate:
            entropy = -(p_hat.clamp_min(1e-8).log() * p_hat).sum(-1) / math.log(self.num_bins)
            gate = 1.0 - torch.sigmoid(self.gate_logit) * entropy.mean(-1, keepdim=True)
            weight = weight * gate
            info["gate"] = gate
        if condition_keep is not None:
            weight = weight * condition_keep.view(-1, 1).to(weight.dtype)
        info.update(p_hat=p_hat, q=q)
        return weight * embedding + (1.0 - weight) * null_embedding, info

    @staticmethod
    def _film(film, feature, embedding):
        return feature if embedding is None else film(feature, embedding)

    def forward(self, image, logits=None, condition_keep=None,
                return_degradation_features=False, return_condition=False):
        image, height, width = self._pad_input(image)
        (enc1, enc2, enc3), latent = self._encode(image)
        if return_degradation_features:
            return [enc1, enc2, enc3], latent

        e, info = self._condition(logits, image.shape[0], image.device, condition_keep)
        dec_film, skip_film = self.decoder_film, self.skip_film
        latent = self._film(self.middle_film, latent, e)

        x = self.up4_3(latent)
        x = self.decoder_level3(self.reduce_chan_level3(torch.cat([x, self._film(skip_film[0], enc3, e)], 1)))
        x = self._film(dec_film[0], x, e)

        x = self.up3_2(x)
        x = self.decoder_level2(self.reduce_chan_level2(torch.cat([x, self._film(skip_film[1], enc2, e)], 1)))
        x = self._film(dec_film[1], x, e)

        x = self.up2_1(x)
        x = self.decoder_level1(torch.cat([x, self._film(skip_film[2], enc1, e)], 1))
        x = self._film(dec_film[2], x, e)
        x = self._film(dec_film[3], self.refinement(x), e)

        if self.scale != 1:
            raise NotImplementedError("scale=1 restoration only")
        output = (self.output(x) + image)[..., :height, :width]
        return (output, info) if return_condition else output
