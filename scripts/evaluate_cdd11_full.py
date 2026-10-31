"""One CDD-11 evaluator for every model (official test: 11 settings x 200 images).

Protocol (same as the DCPT reproduction's SRModel): full image, reflect-pad to a
multiple of 16, RGB, crop_border 0, uint8-rounded PSNR / SSIM.

Modes
  dcpt    plain NAFNet (TLC), e.g. DCPT_NAFNet_CDD11_Finetune_750k checkpoints
  frozen  old Stage 3 / Stage 3a (FrozenFeatureConditionedNAFNet; condition from its own frozen trunk)
  full    new Stage 3b (SeverityConditionedFullNAFNet; condition from the frozen analysis copy)

Ablation flags for --mode full: --shuffle_condition, --no_calibrator, --no_gate, --constant_condition.
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basicsr.archs.feature_severity_extractor_arch import FeatureSeverityExtractor  # noqa: E402
from basicsr.archs.frozen_feature_conditioned_nafnet_arch import FrozenFeatureConditionedNAFNet  # noqa: E402
from basicsr.archs.nafnet_arch import NAFNet  # noqa: E402
from basicsr.archs.severity_conditioned_full_nafnet_arch import SeverityConditionedFullNAFNet  # noqa: E402
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim  # noqa: E402
from basicsr.utils.severity_full_analysis import (  # noqa: E402
    CDD11_SETTINGS, EXTRACTOR_ARGS, NAFNET_ARGS, RESTORMER_ARGS, STAGE2_DIR, AnalysisCopy, cdd11_test_records,
    load_state, read_rgb_tensor,
)

TRIPLES = {"low_haze_rain": "L+H+R", "low_haze_snow": "L+H+S"}


def pad16(image):
    h, w = image.shape[-2:]
    ph, pw = (-h) % 16, (-w) % 16
    return (F.pad(image, (0, pw, 0, ph), mode="reflect") if ph or pw else image), h, w


def count(module):
    return sum(p.numel() for p in module.parameters()) / 1e6


def build(args, device):
    key = args.param_key
    if args.mode == "dcpt":
        net = NAFNet(**NAFNET_ARGS)
        net.load_state_dict(load_state(args.ckpt, key), strict=True)
        net = net.to(device).eval()

        def run(image, row):
            padded, h, w = pad16(image)
            return net(padded)[..., :h, :w]
        return run, {"restorer_M": count(net), "analysis_M": 0.0}

    if args.mode == "frozen":
        net = FrozenFeatureConditionedNAFNet(condition_dim=16, adapter_dim=128, **NAFNET_ARGS)
        net.load_state_dict(load_state(args.ckpt, "params"), strict=True)
        extractor = FeatureSeverityExtractor(**EXTRACTOR_ARGS)
        extractor.load_state_dict(load_state(args.extractor), strict=True)
        net, extractor = net.to(device).eval(), extractor.to(device).eval()

        def run(image, row):
            enc, mid = net(image, return_degradation_features=True)
            condition = extractor(enc, mid).softmax(-1).flatten(1)
            return net(image, condition)
        return run, {"restorer_M": count(net), "analysis_M": count(extractor)}

    full_kwargs = dict(
        adapter_dim=128,
        condition_mode="constant" if args.constant_condition else "full",
        use_gate=not args.no_gate, use_calibrator=not args.no_calibrator,
        logit_temperatures=args.temperatures,
    )
    if args.backbone == "restormer":
        from basicsr.archs.severity_conditioned_full_restormer_arch import SeverityConditionedFullRestormer
        net = SeverityConditionedFullRestormer(**RESTORMER_ARGS, **full_kwargs)
    else:
        net = SeverityConditionedFullNAFNet(**NAFNET_ARGS, **full_kwargs)
    missing, unexpected = net.load_state_dict(load_state(args.ckpt, key), strict=False)
    if missing or unexpected:
        print(f"[WARN] missing={missing} unexpected={unexpected}")
    analysis = AnalysisCopy(args.generator, args.extractor, backbone=args.backbone)
    net, analysis = net.to(device).eval(), analysis.to(device)
    shuffled = {}
    if args.shuffle_condition:
        # Each image receives the analysis logits of a random image from a *different* setting.
        records = cdd11_test_records(args.test_root)
        rng = random.Random(0)
        for row in records:
            donor = rng.choice([r for r in records[::50] if r["setting"] != row["setting"]])
            shuffled[row["id"]] = donor["lq_path"]

    def run(image, row):
        source = image
        if args.shuffle_condition:
            source = read_rgb_tensor(shuffled[row["id"]])[None].to(device)
        logits = analysis(source)
        return net(image, logits)
    return run, {"restorer_M": count(net), "analysis_M": count(analysis)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["dcpt", "frozen", "full"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--backbone", default="nafnet", choices=["nafnet", "restormer"],
                        help="--mode full only; give the matching Stage-2 --generator/--extractor")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--param_key", default="params_ema", help="falls back to params")
    parser.add_argument("--generator", default=str(STAGE2_DIR / "net_g_latest.pth"))
    parser.add_argument("--extractor", default=str(STAGE2_DIR / "net_extractor_latest.pth"))
    parser.add_argument("--test_root", default="/scratch/p26cs0003/krish/datasets/CDD11/test")
    parser.add_argument("--out_dir", default="/scratch/p26cs0003/krish/severity_full/eval")
    parser.add_argument("--limit_per_setting", type=int, default=0, help="debug only")
    parser.add_argument("--shuffle_condition", action="store_true")
    parser.add_argument("--no_calibrator", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--constant_condition", action="store_true", help="evaluate the constant-condition control")
    parser.add_argument("--temperatures", type=float, nargs=4, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run, params = build(args, device)
    records = cdd11_test_records(args.test_root)
    if args.limit_per_setting:
        seen = defaultdict(int)
        records = [r for r in records if (seen.__setitem__(r["setting"], seen[r["setting"]] + 1)
                                          or seen[r["setting"]] <= args.limit_per_setting)]

    psnr, ssim = defaultdict(list), defaultdict(list)
    start = time.time()
    with torch.inference_mode():
        for index, row in enumerate(records):
            image = read_rgb_tensor(row["lq_path"])[None].to(device)
            gt = read_rgb_tensor(row["gt_path"])[None].numpy()
            output = run(image, row).clamp(0, 1).float().cpu().numpy()
            psnr[row["setting"]].append(calculate_psnr(output, gt, 0))
            ssim[row["setting"]].append(calculate_ssim(output, gt, 0))
            if index % 200 == 0:
                print(f"{index}/{len(records)} {row['setting']} psnr={psnr[row['setting']][-1]:.2f}", flush=True)

    table = []
    for setting in CDD11_SETTINGS:
        if psnr[setting]:
            table.append({"setting": setting, "label": TRIPLES.get(setting, setting),
                          "psnr": float(np.mean(psnr[setting])), "ssim": float(np.mean(ssim[setting])),
                          "n": len(psnr[setting])})
    summary = {
        "tag": args.tag, "mode": args.mode, "ckpt": args.ckpt, "flags": {
            k: getattr(args, k) for k in ("shuffle_condition", "no_calibrator", "no_gate", "constant_condition", "temperatures")},
        "avg_psnr": float(np.mean([r["psnr"] for r in table])),
        "avg_ssim": float(np.mean([r["ssim"] for r in table])),
        "seconds_per_image": (time.time() - start) / max(len(records), 1),
        **params, "per_setting": table,
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"{args.tag}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["setting", "label", "psnr", "ssim", "n"])
        writer.writeheader()
        writer.writerows(table)
    print(f"\n== {args.tag} ({args.mode}) ==")
    for r in table:
        print(f"{r['label']:>15s}  {r['psnr']:6.2f}  {r['ssim']:.4f}")
    print(f"{'11-cat avg':>15s}  {summary['avg_psnr']:6.2f}  {summary['avg_ssim']:.4f}")
    print(f"params: restorer {params['restorer_M']:.1f}M + analysis {params['analysis_M']:.1f}M")
    print(f"saved: {out / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
