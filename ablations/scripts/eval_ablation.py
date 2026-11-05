"""Evaluate one Stage 3 ablation checkpoint (or a DCPT checkpoint) with optional test-time probes.

Test sets
  cdd11   official CDD-11 test, 11 settings x 200 images (same protocol as scripts/evaluate_cdd11_full.py)
  graded  severity-graded set from ablations/scripts/build_graded_testset.py (levels 1-3 in range, 4 beyond)

Probes: test-time changes to the condition only; the trained network is never changed.
  base          condition from the frozen analysis copy (normal use)
  shuffle       logits of a random image from a different setting / composition      (C3)
  null          the learned null condition for every image                           (C3)
  sev_up        severity of every primitive moved one bin up (clamped)              (C3)
  sev_down      ... one bin down                                                     (C3)
  noise<S>      Gaussian noise with std S (logit units) on the logits, e.g. noise1.0 (C4)
  oracle_type   true presence from the setting name, predicted severity kept         (S3)
  oracle_full   true presence and true severity bin (graded set only)                (S3)

Per image it records PSNR (+ SSIM for the base probe unless --ssim_all), the gate value,
the mean normalized entropy of the condition, and the predicted severity bins (K6 / X1).

Output: <out_dir>/<tag>/<testset>__<probe>.json (summary + per-image rows).
"""
import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ablations.register  # noqa: E402,F401
from ablations.ablib import conditions as cond  # noqa: E402
from ablations.ablib.common import (  # noqa: E402
    CDD11_SETTINGS, DEFAULT_TEST_ROOT, GROUPS, NAFNET_ARGS, analysis_from_config, build_network, cdd11_test_records,
    image_metrics, load_config, load_state, mean, pad16, read_jsonl, read_rgb_tensor, write_json,
)
from basicsr.archs.nafnet_arch import NAFNet  # noqa: E402

CONDITION_PROBES = ("shuffle", "null", "sev_up", "sev_down", "oracle_type", "oracle_full")


def parse_probe(name):
    if name == "base" or name in CONDITION_PROBES:
        return name, None
    if name.startswith("noise"):
        return "noise", float(name[len("noise"):])
    raise ValueError(f"Unknown probe {name}")


def build(args, device):
    """Returns (restore(image, logits) -> (output, info), analysis or None, description)."""
    if args.mode == "dcpt":
        net = NAFNet(**NAFNET_ARGS)
        net.load_state_dict(load_state(args.ckpt, args.param_key), strict=True)
        net = net.to(device).eval()

        def restore(image, logits):
            padded, h, w = pad16(image)
            return net(padded)[..., :h, :w], {}
        return restore, None, {"mode": "dcpt", "test_transform": None, "net": net}

    opt = load_config(args.config)
    network_opt = dict(opt["network_g"])
    network_opt.pop("shared_extractor_path", None)  # weights come from the checkpoint
    net = build_network(network_opt)
    missing, unexpected = net.load_state_dict(load_state(args.ckpt, args.param_key), strict=False)
    if missing or unexpected:
        print(f"[WARN] missing={missing} unexpected={unexpected}", flush=True)
    net = net.to(device).eval()
    shared = getattr(net, "analysis", "frozen") == "shared"
    analysis = None if shared else analysis_from_config(opt).to(device)
    test_transform = (opt.get("ablation") or {}).get("test_condition_transform")

    def restore(image, logits):
        output, info = net(image, logits, return_condition=True)
        return output, info
    return restore, analysis, {"mode": "config", "test_transform": test_transform, "net": net, "shared": shared}


def load_records(testset, args):
    if testset == "cdd11":
        rows = cdd11_test_records(args.test_root)
        for row in rows:
            row["group"] = row["setting"]
        order = list(CDD11_SETTINGS)
    else:
        rows = read_jsonl(args.graded_manifest)
        for row in rows:
            row["group"] = f"{row['composition']}@L{row['level']}"
        order = sorted({r["group"] for r in rows}, key=lambda g: (g.split("@")[0], g.split("@")[1]))
    if args.limit_per_setting:
        seen = defaultdict(int)
        kept = []
        for row in rows:
            seen[row["group"]] += 1
            if seen[row["group"]] <= args.limit_per_setting:
                kept.append(row)
        rows = kept
    return rows, order


def condition_info(info):
    out = {"gate": None, "entropy": None, "pred_bins": None, "presence": None}
    p_hat = info.get("p_hat") if info else None
    if p_hat is None:
        return out
    probs = p_hat[0].float()
    entropy = -(probs.clamp_min(1e-8).log() * probs).sum(-1) / np.log(probs.shape[-1])
    out["entropy"] = float(entropy.mean())
    out["pred_bins"] = (probs[:, 1:].argmax(-1) + 1).tolist()
    out["presence"] = [round(float(v), 4) for v in (1.0 - probs[:, 0])]
    if info.get("gate") is not None:
        out["gate"] = float(info["gate"].mean())
    return out


def summarize(rows, order, testset):
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    table = []
    for group in order:
        items = by_group.get(group)
        if not items:
            continue
        table.append({"group": group, "n": len(items), "psnr": mean([r["psnr"] for r in items]),
                      "ssim": mean([r["ssim"] for r in items]), "gate": mean([r["gate"] for r in items]),
                      "entropy": mean([r["entropy"] for r in items])})
    summary = {"avg_psnr": mean([t["psnr"] for t in table]), "avg_ssim": mean([t["ssim"] for t in table])}
    if testset == "cdd11":
        for name, settings in GROUPS.items():
            summary[f"{name}_psnr"] = mean([t["psnr"] for t in table if t["group"] in settings])
    else:
        summary["in_range_psnr"] = mean([t["psnr"] for t in table if not t["group"].endswith("@L4")])
        summary["beyond_psnr"] = mean([t["psnr"] for t in table if t["group"].endswith("@L4")])
        for level in (1, 2, 3, 4):
            summary[f"L{level}_psnr"] = mean([t["psnr"] for t in table if t["group"].endswith(f"@L{level}")])
    summary["gate_mean"] = mean([r["gate"] for r in rows])
    summary["entropy_mean"] = mean([r["entropy"] for r in rows])
    return summary, table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="config", choices=["config", "dcpt"])
    parser.add_argument("--config", help="RENDERED Stage 3 ablation config (architecture + analysis copy)")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--param_key", default="params_ema", help="falls back to params")
    parser.add_argument("--testsets", nargs="+", default=["cdd11"], choices=["cdd11", "graded"])
    parser.add_argument("--graded_manifest", default=None)
    parser.add_argument("--probes", nargs="+", default=["base"])
    parser.add_argument("--test_root", default=DEFAULT_TEST_ROOT)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--limit_per_setting", type=int, default=0, help="debug / quick look only")
    parser.add_argument("--ssim_all", action="store_true", help="also compute SSIM for non-base probes (slow)")
    args = parser.parse_args()
    if args.mode == "config" and not args.config:
        parser.error("--config is required unless --mode dcpt")
    if "graded" in args.testsets and not args.graded_manifest:
        parser.error("--graded_manifest is required for the graded test set")
    probes = [parse_probe(p) for p in args.probes]
    if args.mode == "dcpt" and any(kind != "base" for kind, _ in probes):
        parser.error("DCPT has no condition: only the base probe applies")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    restore, analysis, meta = build(args, device)
    if meta.get("shared") and any(kind != "base" for kind, _ in probes):
        parser.error("the shared-analysis model (S4) computes its own condition: only the base probe applies")
    out_root = Path(args.out_dir) / args.tag

    for testset in args.testsets:
        rows, order = load_records(testset, args)
        probe_list = [(k, v) for k, v in probes if not (k == "oracle_full" and testset != "graded")]
        results = {f"{k}{'' if v is None else v}": [] for k, v in probe_list}
        donors = [r for r in rows[::max(1, len(rows) // 220)]]
        donor_logits = {}
        rng = random.Random(0)
        start = time.time()
        with torch.inference_mode():
            for index, row in enumerate(rows):
                image = read_rgb_tensor(row["lq_path"])[None].to(device)
                gt = read_rgb_tensor(row["gt_path"])[None].numpy()
                logits = None
                if analysis is not None:
                    logits = analysis(image)
                    if meta["test_transform"]:
                        logits = cond.apply_transform(meta["test_transform"], logits, row["group"])
                for kind, value in probe_list:
                    probe_logits, constant = logits, False
                    if kind == "shuffle":
                        candidates = [d for d in donors if d["group"].split("@")[0] != row["group"].split("@")[0]]
                        donor = rng.choice(candidates)
                        if donor["id"] not in donor_logits:
                            donor_image = read_rgb_tensor(donor["lq_path"])[None].to(device)
                            donor_value = analysis(donor_image)
                            if meta["test_transform"]:
                                donor_value = cond.apply_transform(meta["test_transform"], donor_value, donor["group"])
                            donor_logits[donor["id"]] = donor_value
                        probe_logits = donor_logits[donor["id"]]
                    elif kind == "null":
                        constant = True
                    elif kind == "sev_up":
                        probe_logits = cond.shift_severity(logits, +1)
                    elif kind == "sev_down":
                        probe_logits = cond.shift_severity(logits, -1)
                    elif kind == "noise":
                        probe_logits = cond.add_noise(logits, value, seed=index)
                    elif kind == "oracle_type":
                        probe_logits = cond.oracle_type(logits, cond.active_from_setting(row["group"], device))
                    elif kind == "oracle_full":
                        probe_logits = cond.oracle_full(row["severity_bins"], device=device)[None]
                    net = meta["net"]
                    if constant:
                        previous, net.condition_mode = net.condition_mode, "constant"
                    try:
                        output, info = restore(image, probe_logits)
                    finally:
                        if constant:
                            net.condition_mode = previous
                    with_ssim = kind == "base" or args.ssim_all
                    psnr, ssim = image_metrics(output, gt, with_ssim)
                    record = {"id": row["id"], "group": row["group"], "psnr": psnr, "ssim": ssim}
                    if testset == "graded":
                        record.update(level=row["level"], composition=row["composition"])
                    record.update(condition_info(info))
                    results[f"{kind}{'' if value is None else value}"].append(record)
                if index % 200 == 0:
                    base = results[next(iter(results))][-1]
                    print(f"[{testset}] {index}/{len(rows)} {row['group']} psnr={base['psnr']:.2f}", flush=True)
        seconds = (time.time() - start) / max(len(rows), 1)
        for probe_name, records in results.items():
            summary, table = summarize(records, order, testset)
            payload = {"tag": args.tag, "testset": testset, "probe": probe_name, "mode": meta["mode"],
                       "config": args.config, "ckpt": args.ckpt, "param_key": args.param_key,
                       "test_condition_transform": meta["test_transform"], "seconds_per_image_all_probes": seconds,
                       **summary, "per_group": table, "per_image": records}
            path = out_root / f"{testset}__{probe_name}.json"
            write_json(path, payload)
            print(f"== {args.tag} {testset} {probe_name}: avg PSNR {summary['avg_psnr']:.3f}"
                  + (f" SSIM {summary['avg_ssim']:.4f}" if summary["avg_ssim"] else "") + f" -> {path}", flush=True)


if __name__ == "__main__":
    main()
