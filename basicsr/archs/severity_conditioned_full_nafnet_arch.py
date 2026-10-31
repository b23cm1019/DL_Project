"""Stage 3b restorer: fully trainable NAFNet conditioned on calibrated severity.

Differences from ``FrozenFeatureConditionedNAFNet`` (Stage 3a):
  * every parameter is trainable (the owning model decides; nothing is frozen here);
  * the network takes the frozen extractor's raw logits ``z`` [B, 4, 4] and
    computes the condition itself:  z -> calibrator -> p_hat -> projector -> e;
  * a learned confidence gate g = 1 - lambda * mean normalized entropy blends the
    condition embedding toward a learned null condition (lambda starts ~0);
  * condition dropout also swaps in the null condition (FiLM is never removed,
    because the Stage-3a FiLM learned scale ~ -0.95 and the decoder relies on it);
  * zero-initialized FiLM is added on the four encoder skip connections;
  * TLC test-time local pooling (``Local_Base``), exactly like DCPT's ``NAFNet``.

Parameter names of intro/encoders/downs/middle_blks/ups/decoder*/ending,
``condition_projector``, ``middle_film`` and ``decoder_film`` match Stage 3a,
so a Stage 3a checkpoint loads directly; only ``skip_film`` and ``calibrator``
are new and both start as identity maps.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from basicsr.archs.arch_util import Local_Base
from basicsr.archs.frozen_feature_conditioned_nafnet_arch import _SeverityFiLM
from basicsr.archs.nafnet_arch import NAFNetBaseline
from basicsr.utils.registry import ARCH_REGISTRY


class SeverityCalibrator(nn.Module):
    """Restoration-aware ordinal severity calibrator (16 parameters for 4 types).

    Per primitive t:  presence  pi_t = 1 - q_t0 is kept fixed (type cannot change),
    severity logits   s_t = alpha_t * z_t[1:] + b_t  are re-weighted,
    p_hat_t = [q_t0, pi_t * softmax(s_t)].  At init (alpha=1, b=0) p_hat == softmax(z).
    """

    def __init__(self, num_types=4, num_bins=4):
        super().__init__()
        # softplus(raw) == 1 at init.
        self.raw_alpha = nn.Parameter(torch.full((num_types, 1), math.log(math.e - 1.0)))
        self.bias = nn.Parameter(torch.zeros(num_types, num_bins - 1))

    def forward(self, logits, enabled=True):
        frozen = logits.softmax(dim=-1)
        if not enabled:
            return frozen, frozen
        alpha = F.softplus(self.raw_alpha)
        severity = (alpha * logits[..., 1:] + self.bias).softmax(dim=-1)
        absent = frozen[..., :1]
        calibrated = torch.cat([absent, (1.0 - absent) * severity], dim=-1)
        return calibrated, frozen


@ARCH_REGISTRY.register()
class SeverityConditionedFullNAFNet(Local_Base, NAFNetBaseline):
    def __init__(
        self,
        *args,
        num_types=4,
        num_bins=4,
        adapter_dim=128,
        # "full": per-image calibrated p.  "constant": always the learned null
        # condition (control: same arch/init, no per-image information).
        # "none": FiLM removed (only valid when NOT warm-starting from Stage 3a).
        condition_mode="full",
        use_gate=True,
        gate_init=-6.0,  # lambda = sigmoid(gate_init) ~ 0.0025: warm start is unchanged
        use_calibrator=True,
        logit_temperatures=None,  # optional post-hoc T per primitive (plan section A)
        train_size=(1, 3, 128, 128),
        fast_imp=False,
        **kwargs,
    ):
        Local_Base.__init__(self)
        NAFNetBaseline.__init__(self, *args, prompt_dim=0, **kwargs)
        if condition_mode not in {"full", "constant", "none"}:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")
        self.condition_mode = condition_mode
        self.use_gate = use_gate
        self.use_calibrator = use_calibrator
        self.num_types, self.num_bins = num_types, num_bins
        condition_dim = num_types * num_bins
        temperatures = torch.ones(num_types) if logit_temperatures is None else torch.tensor(
            [float(t) for t in logit_temperatures])
        # Non-persistent: the value always comes from the config, not the checkpoint.
        self.register_buffer("logit_temperatures", temperatures.view(1, num_types, 1), persistent=False)

        width = self.intro.out_channels
        levels = len(self.encoders)
        self.calibrator = SeverityCalibrator(num_types, num_bins)
        # Learned "null" condition (uniform p at init).  Used for condition
        # dropout, for the gate's fallback and for the constant-condition control.
        self.null_logits = nn.Parameter(torch.zeros(num_types, num_bins))
        # Learned confidence gate  g = 1 - sigmoid(gate_logit) * mean normalized entropy.
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, adapter_dim), nn.GELU(), nn.Linear(adapter_dim, adapter_dim)
        )
        self.middle_film = _SeverityFiLM(adapter_dim, width * (2 ** levels))
        decoder_channels = [width * (2 ** level) for level in reversed(range(levels))]
        self.decoder_film = nn.ModuleList([_SeverityFiLM(adapter_dim, c) for c in decoder_channels])
        self.skip_film = nn.ModuleList([_SeverityFiLM(adapter_dim, c) for c in decoder_channels])

        # TLC conversion runs one dummy forward; it must work without logits.
        _, _, height, width_px = train_size
        self.eval()
        with torch.no_grad():
            self.convert(
                base_size=(int(height * 1.5), int(width_px * 1.5)),
                train_size=train_size,
                fast_imp=fast_imp,
            )
        self.train()

    # ------------------------------------------------------------------ helpers
    def _condition(self, logits, batch, device, condition_keep=None):
        """Return (embedding [B, D], info dict).

        e = keep * [g * e(p_hat) + (1 - g) * e(p_null)] + (1 - keep) * e(p_null)
        The gate and dropout blend toward a learned null condition instead of
        removing FiLM, because the Stage-3a decoder depends on FiLM being present.
        """
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
        if condition_keep is not None:  # condition dropout (training only)
            weight = weight * condition_keep.view(-1, 1).to(weight.dtype)
        info.update(p_hat=p_hat, q=q)
        return weight * embedding + (1.0 - weight) * null_embedding, info

    @staticmethod
    def _film(film, feature, embedding):
        if embedding is None:
            return feature
        return film(feature, embedding)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        image,
        logits=None,
        condition_keep=None,
        return_degradation_features=False,
        return_condition=False,
    ):
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
        if return_degradation_features:
            return skips, self.middle_blks(x)

        embedding, info = self._condition(logits, image.shape[0], image.device, condition_keep)
        x = self._film(self.middle_film, self.middle_blks(x), embedding)
        for index, (up, skip, dec_film, skip_film) in enumerate(
            zip(self.ups, reversed(skips), self.decoder_film, self.skip_film)
        ):
            x = up(x) + self._film(skip_film, skip, embedding)
            x = self._film(dec_film, getattr(self, f"decoder{index}")(x), embedding)
        output = (self.ending(x) + image)[..., :height, :width]
        if return_condition:
            return output, info
        return output
