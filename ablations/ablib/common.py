"""Shared helpers for the ablation scripts: config reading, analysis copies, records, metrics."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from basicsr.archs import build_network
from basicsr.archs.nafnet_arch import NAFNet
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.utils.options import yaml_load
from basicsr.utils.severity_full_analysis import (
    CDD11_SETTINGS, NAFNET_ARGS, STAGE2_DIR, TYPES, AnalysisCopy, cdd11_test_records, load_state, read_rgb_tensor,
)

GROUPS = {
    "single": ("low", "haze", "rain", "snow"),
    "double": ("low_haze", "low_rain", "low_snow", "haze_rain", "haze_snow"),
    "triple": ("low_haze_rain", "low_haze_snow"),
}
DEFAULT_TEST_ROOT = "/scratch/p26cs0003/krish/datasets/CDD11/test"

__all__ = [
    "CDD11_SETTINGS", "GROUPS", "NAFNET_ARGS", "TYPES", "AblationAnalysisCopy", "analysis_from_config",
    "build_analysis", "cdd11_test_records", "encoder_features", "image_metrics", "load_config", "load_state",
    "pad16", "read_jsonl", "read_rgb_tensor", "write_json", "write_jsonl",
]


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def load_config(path):
    text = Path(path).read_text()
    if "@OUT_ROOT@" in text or "@SHARED_ROOT@" in text:
        raise ValueError(f"{path} still has @OUT_ROOT@/@SHARED_ROOT@: pass the RENDERED config "
                         "($OUT_ROOT/ablations/configs/<name>.yml, written by the ablation launchers)")
    return yaml_load(str(path))


def pad16(image):
    h, w = image.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    return (F.pad(image, (0, pw, 0, ph), mode="reflect") if ph or pw else image), h, w


class AblationAnalysisCopy(torch.nn.Module):
    """Frozen Stage 2 NAFNet trunk + an ablation extractor variant (S5 / S10 / T5 / X3)."""

    def __init__(self, generator_path, extractor_path, extractor_net):
        super().__init__()
        self.trunk = NAFNet(**NAFNET_ARGS)
        self.extractor = build_network(dict(extractor_net))
        self.trunk.load_state_dict(load_state(generator_path), strict=True)
        self.extractor.load_state_dict(load_state(extractor_path), strict=True)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, image):
        encoder_features, middle = self.trunk(image, return_degradation_features=True)
        return self.extractor(encoder_features, middle)


def build_analysis(generator=None, extractor=None, extractor_net=None):
    generator = generator or str(STAGE2_DIR / "net_g_latest.pth")
    extractor = extractor or str(STAGE2_DIR / "net_extractor_latest.pth")
    if extractor_net:
        return AblationAnalysisCopy(generator, extractor, extractor_net)
    return AnalysisCopy(generator, extractor, backbone="nafnet")


def analysis_from_config(opt):
    """Analysis copy named in a Stage 3 ablation config (``ablation.analysis``)."""
    spec = (opt.get("ablation") or {}).get("analysis") or {}
    return build_analysis(spec.get("generator"), spec.get("extractor"), spec.get("extractor_net"))


def encoder_features(net, image):
    """Unconditioned encoder + middle features of any NAFNet-layout network, unpadded.

    This is the exact path the frozen analysis copy uses, so features of a fine-tuned restorer
    can be compared with those of the frozen Stage 2 trunk on the same input (S2).
    """
    x = net.intro(image)
    skips = []
    for encoder, down in zip(net.encoders, net.downs):
        x = encoder(x)
        skips.append(x)
        x = down(x)
    return skips, net.middle_blks(x)


def image_metrics(output, gt, with_ssim=True):
    """PSNR / SSIM exactly as scripts/evaluate_cdd11_full.py computes them."""
    output = output.clamp(0, 1).float().cpu().numpy()
    psnr = float(calculate_psnr(output, gt, 0))
    ssim = float(calculate_ssim(output, gt, 0)) if with_ssim else None
    return psnr, ssim


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None
