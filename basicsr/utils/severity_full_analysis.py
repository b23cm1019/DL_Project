"""Frozen analysis copy for the Stage 3b pipeline (shared by caching and evaluation).

It reproduces exactly the path on which the Stage-2 extractor was validated:
TLC ``NAFNet`` trunk -> unpadded full image -> encoder/middle features ->
``FeatureSeverityExtractor`` -> logits [B, 4, 4].
"""
from pathlib import Path

import cv2
import torch

from basicsr.archs.feature_severity_extractor_arch import FeatureSeverityExtractor
from basicsr.archs.nafnet_arch import NAFNet

TYPES = ("low", "haze", "rain", "snow")
CDD11_SETTINGS = (
    "low", "haze", "rain", "snow", "low_haze", "low_rain", "low_snow",
    "haze_rain", "haze_snow", "low_haze_rain", "low_haze_snow",
)
NAFNET_ARGS = dict(width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1,
                   dec_blk_nums=[1, 1, 1, 1], window_size=16)
EXTRACTOR_ARGS = dict(feature_dims=[64, 128, 256, 512], middle_dim=1024, hidden_dim=256,
                      num_types=4, num_bins=4)
STAGE2_DIR = Path("/scratch/p26cs0003/krish/experiments/SeverityDCPT_Stage2_CDDAdapt/models")

# Restormer arm (same configuration as options/cdd_experiments/severity_restormer_*.yml)
RESTORMER_ARGS = dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                      num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                      bias=False, LayerNorm_type="WithBias", dual_pixel_task=False, scale=1,
                      window_size=8)
RESTORMER_EXTRACTOR_ARGS = dict(feature_dims=[48, 96, 192], middle_dim=384, hidden_dim=256,
                                num_types=4, num_bins=4)


def backbone_specs(backbone):
    """(trunk class, trunk kwargs, extractor kwargs) for a supported backbone."""
    if backbone == "nafnet":
        return NAFNet, NAFNET_ARGS, EXTRACTOR_ARGS
    if backbone == "restormer":
        from basicsr.archs.feature_tap_restormer_arch import FeatureTapRestormer
        return FeatureTapRestormer, RESTORMER_ARGS, RESTORMER_EXTRACTOR_ARGS
    raise ValueError(f"Unknown backbone: {backbone}")


def load_state(path, key="params"):
    state = torch.load(path, map_location="cpu")
    if key in state:
        state = state[key]
    elif "params" in state:
        state = state["params"]
    return {k.removeprefix("module."): v for k, v in state.items()}


def read_rgb_tensor(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1)).contiguous()


class AnalysisCopy(torch.nn.Module):
    """Frozen Stage-2 trunk + extractor. Never trained after Stage 2."""

    def __init__(self, generator_path=STAGE2_DIR / "net_g_latest.pth",
                 extractor_path=STAGE2_DIR / "net_extractor_latest.pth", backbone="nafnet"):
        super().__init__()
        trunk_cls, trunk_args, extractor_args = backbone_specs(backbone)
        self.trunk = trunk_cls(**trunk_args)
        self.extractor = FeatureSeverityExtractor(**extractor_args)
        self.trunk.load_state_dict(load_state(generator_path), strict=True)
        self.extractor.load_state_dict(load_state(extractor_path), strict=True)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    def train(self, mode=True):  # always eval
        return super().train(False)

    @torch.no_grad()
    def forward(self, image):
        encoder_features, middle = self.trunk(image, return_degradation_features=True)
        return self.extractor(encoder_features, middle)


def cdd11_test_records(root="/scratch/p26cs0003/krish/datasets/CDD11/test"):
    root = Path(root)
    records = []
    for setting in CDD11_SETTINGS:
        for lq_path in sorted((root / setting).glob("*.png")):
            records.append({
                "id": f"test/{setting}/{lq_path.name}", "setting": setting,
                "lq_path": str(lq_path), "gt_path": str(root / "clear" / lq_path.name),
            })
    return records
