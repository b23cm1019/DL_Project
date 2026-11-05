"""Condition helpers shared by training (datasets) and evaluation (probes).

All functions take and return severity logits of shape [..., T, K] (T = 4 primitives
low/haze/rain/snow, K = 4 bins: bin 0 = absent, bins 1..3 = severity levels). They return log
probabilities, which the restorer treats exactly like raw logits (it applies a softmax).
"""
import torch

from basicsr.utils.severity_full_analysis import TYPES

EPS = 1e-3  # probability given to "impossible" bins in oracle conditions


def _logits(probs):
    return probs.clamp_min(1e-8).log()


def active_from_setting(setting, device=None):
    """CDD-11 setting / graded composition name -> bool tensor [T] (which primitives are present)."""
    names = set(str(setting).split("@")[0].split("_"))
    return torch.tensor([name in names for name in TYPES], dtype=torch.bool, device=device)


def type_only(logits):
    """Keep P(absent) per primitive; spread the present mass evenly over the severity bins.

    The calibrator then sees identical severity logits for every image, so no per-image severity
    information reaches the restorer (S1: does severity help beyond the type?).
    """
    probs = logits.softmax(-1)
    absent = probs[..., :1]
    bins = probs.shape[-1] - 1
    present = (1.0 - absent).expand(*absent.shape[:-1], bins) / bins
    return _logits(torch.cat([absent, present], -1))


def oracle_type(logits, active):
    """True presence (from the setting name), predicted severity kept (S3, CDD-11)."""
    conditional = logits[..., 1:].softmax(-1)
    absent = torch.where(active, torch.full_like(conditional[..., 0], EPS),
                         torch.full_like(conditional[..., 0], 1.0 - EPS))[..., None]
    return _logits(torch.cat([absent, (1.0 - absent) * conditional], -1))


def oracle_full(bins, num_bins=4, device=None):
    """True presence and true severity bin (S3, graded set). ``bins``: long [..., T], 0 = absent."""
    bins = torch.as_tensor(bins, device=device).long().clamp(0, num_bins - 1)
    probs = torch.full((*bins.shape, num_bins), EPS, device=bins.device)
    probs.scatter_(-1, bins[..., None], 1.0 - (num_bins - 1) * EPS)
    return _logits(probs)


def set_type_level(logits, type_index, level, num_bins=4):
    """Overwrite one primitive's distribution: level 0 = absent, 1..3 = that severity (X2)."""
    out = logits.clone()
    probs = torch.full((num_bins,), EPS, device=logits.device)
    probs[level] = 1.0 - (num_bins - 1) * EPS
    out[..., type_index, :] = _logits(probs)
    return out


def shift_severity(logits, delta):
    """Move the severity distribution of every primitive by ``delta`` bins (clamped) (C3)."""
    probs = logits.softmax(-1)
    absent = probs[..., :1]
    conditional = logits[..., 1:].softmax(-1)
    bins = conditional.shape[-1]
    shifted = torch.zeros_like(conditional)
    for index in range(bins):
        target = min(max(index + delta, 0), bins - 1)
        shifted[..., target] += conditional[..., index]
    return _logits(torch.cat([absent, (1.0 - absent) * shifted], -1))


def add_noise(logits, sigma, seed):
    """Gaussian noise (std ``sigma``, in logit units) with a fixed seed per image (C4)."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(logits.shape, generator=generator).to(logits.device, logits.dtype)
    return logits + sigma * noise


def apply_transform(name, logits, setting=None):
    """Named transform used by datasets (training) and by the evaluator (test time)."""
    if name in (None, "", "none"):
        return logits
    if name == "type_only":
        return type_only(logits)
    if name == "oracle_type":
        return oracle_type(logits, active_from_setting(setting, logits.device))
    raise ValueError(f"Unknown condition transform: {name}")
