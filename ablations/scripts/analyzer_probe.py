"""Analyzer accuracy (S9) and feature-drift diagnostic (S2).

S9  Accuracy of the frozen analysis copy on its own:
      synthetic validation (LSDIR renders, exact bins), severity-graded CDD-11 set (exact bins,
      level 4 = beyond the training range), CDD-11 test (presence labels only).
    Reports severity accuracy / mean absolute error / 3x3 confusion per primitive, expected
    calibration error (ECE), type F1 / exact match, and the mean predicted severity per graded
    level (should rise monotonically, and saturate at bin 3 for level 4).

S2  Drift: the same frozen Stage 2 extractor applied to the encoder features of fine-tuned
    Stage 3 restorers (e.g. Plan B at 50k / 100k / 200k) instead of the frozen trunk.
    If accuracy drops and the features move away from the frozen trunk's, the restorer's own
    encoder can no longer be read for severity -> the analysis path must stay frozen.

Example:
  python ablations/scripts/analyzer_probe.py --config <rendered s3 config> \
      --graded_manifest $OUT_ROOT/ablations/graded_cdd11/manifest.jsonl \
      --restorer_ckpts /path/net_g_50000.pth /path/net_g_100000.pth /path/net_g_200000.pth \
      --out $OUT_ROOT/ablations/probes/analyzer_planb.json
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ablations.register  # noqa: E402,F401
from ablations.ablib.common import (  # noqa: E402
    DEFAULT_TEST_ROOT, TYPES, analysis_from_config, build_analysis, build_network, cdd11_test_records,
    encoder_features, load_config, load_state, read_jsonl, read_rgb_tensor, write_json,
)

SYNTHETIC_VAL = "/scratch/p26cs0003/krish/severity_cdd11_lsdir_outdoor_v1/data/manifests/val.jsonl"


def subsample(rows, limit):
    if limit and len(rows) > limit:
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)]
    return rows


def gather(rows, kind):
    """Rows -> list of dicts with lq path, true bins (or None) and true presence."""
    items = []
    for row in rows:
        if kind == "synthetic":
            bins = row["severity_bins"]
            bins = [bins[t] for t in TYPES] if isinstance(bins, dict) else list(bins)
            items.append({"path": row["image_path"], "bins": bins, "level": None,
                          "active": [b > 0 for b in bins]})
        elif kind == "graded":
            items.append({"path": row["lq_path"], "bins": row["severity_bins"], "level": row["level"],
                          "active": [t in row["active_degradations"] for t in TYPES]})
        else:
            names = set(row["setting"].split("_"))
            items.append({"path": row["lq_path"], "bins": None, "level": None, "active": [t in names for t in TYPES]})
    return items


def metrics(logits, items, num_bins=4, ece_bins=15):
    """logits: [N, T, K] tensor. Severity metrics use only present primitives with known bins
    in 1..3; beyond-range (level 4) graded items are reported separately."""
    probs = logits.softmax(-1)
    conditional = logits[..., 1:].softmax(-1)
    presence = 1.0 - probs[..., 0]
    out = {"n": len(items)}
    true_active = torch.tensor([it["active"] for it in items])
    pred_active = presence >= 0.5
    tp = (pred_active & true_active).sum().item()
    fp = (pred_active & ~true_active).sum().item()
    fn = (~pred_active & true_active).sum().item()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    out["type_f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
    out["type_exact_match"] = (pred_active == true_active).all(1).float().mean().item()

    labeled = [i for i, it in enumerate(items) if it["bins"] is not None and (it["level"] in (None, 1, 2, 3))]
    if labeled:
        per_type, conf_all, correct_all, hits = {}, [], [], []
        for t, name in enumerate(TYPES):
            idx = [i for i in labeled if items[i]["bins"][t] > 0]
            if not idx:
                continue
            target = torch.tensor([items[i]["bins"][t] for i in idx])
            cond_t = conditional[idx, t]
            pred = cond_t.argmax(-1) + 1
            confusion = np.zeros((num_bins - 1, num_bins - 1), dtype=int)
            for a, b in zip(target.tolist(), pred.tolist()):
                confusion[a - 1, b - 1] += 1
            per_type[name] = {"n": len(idx), "accuracy": (pred == target).float().mean().item(),
                              "mae": (pred - target).abs().float().mean().item(), "confusion_true_by_pred": confusion.tolist()}
            conf_all.append(cond_t.max(-1).values)
            correct_all.append((pred == target).float())
            hits.extend((pred == target).tolist())
        confidence, correct = torch.cat(conf_all), torch.cat(correct_all)
        edges = torch.linspace(0, 1, ece_bins + 1)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (confidence > lo) & (confidence <= hi)
            if mask.any():
                ece += mask.float().mean().item() * abs(confidence[mask].mean().item() - correct[mask].mean().item())
        out.update(severity_accuracy=float(np.mean(hits)), ece=ece, per_type=per_type)

    levels = sorted({it["level"] for it in items if it["level"] is not None})
    if levels:
        positions = torch.arange(1, num_bins, dtype=conditional.dtype)
        expected = (conditional * positions).sum(-1)
        curve = {}
        for name_index, name in enumerate(TYPES):
            curve[name] = {}
            for level in levels:
                idx = [i for i, it in enumerate(items) if it["level"] == level and it["active"][name_index]]
                if idx:
                    curve[name][f"L{level}"] = {
                        "mean_expected_bin": expected[idx, name_index].mean().item(),
                        "fraction_pred_bin3": (conditional[idx, name_index].argmax(-1) == 2).float().mean().item(),
                        "n": len(idx)}
        out["graded_level_curve"] = curve
    return out


@torch.no_grad()
def run_frozen(analysis, items, device):
    logits, pooled = [], []
    for item in items:
        image = read_rgb_tensor(item["path"])[None].to(device)
        skips, middle = encoder_features(analysis.trunk, image)
        logits.append(analysis.extractor(skips, middle)[0].float().cpu())
        pooled.append([f.mean((2, 3))[0].float().cpu() for f in list(skips) + [middle]])
    return torch.stack(logits), pooled


@torch.no_grad()
def run_drift(net, extractor, items, frozen_pooled, device):
    logits, cosine, rel = [], defaultdict(list), defaultdict(list)
    for item, reference in zip(items, frozen_pooled):
        image = read_rgb_tensor(item["path"])[None].to(device)
        skips, middle = encoder_features(net, image)
        logits.append(extractor(skips, middle)[0].float().cpu())
        for level, (feature, ref) in enumerate(zip(list(skips) + [middle], reference)):
            vector = feature.mean((2, 3))[0].float().cpu()
            cosine[level].append(torch.nn.functional.cosine_similarity(vector, ref, dim=0).item())
            rel[level].append(((vector - ref).norm() / ref.norm().clamp_min(1e-8)).item())
    names = ["enc1", "enc2", "enc3", "enc4", "middle"]
    similarity = {names[k]: {"pooled_cosine": float(np.mean(cosine[k])), "pooled_relative_l2": float(np.mean(rel[k]))}
                  for k in cosine}
    return torch.stack(logits), similarity


def iteration_of(path):
    match = re.search(r"net_g_(\d+)\.pth$", str(path))
    return int(match.group(1)) if match else str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="RENDERED Stage 3 config: gives the analysis copy and the restorer architecture")
    parser.add_argument("--generator", help="instead of --config: Stage 2 net_g")
    parser.add_argument("--extractor", help="instead of --config: Stage 2 net_extractor")
    parser.add_argument("--graded_manifest")
    parser.add_argument("--synthetic_val", default=SYNTHETIC_VAL)
    parser.add_argument("--synthetic_limit", type=int, default=1500)
    parser.add_argument("--cdd11_limit_per_setting", type=int, default=50)
    parser.add_argument("--test_root", default=DEFAULT_TEST_ROOT)
    parser.add_argument("--restorer_ckpts", nargs="*", default=[], help="S2: fine-tuned Stage 3 checkpoints")
    parser.add_argument("--param_key", default="params_ema")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = load_config(args.config) if args.config else None
    analysis = (analysis_from_config(opt) if opt else build_analysis(args.generator, args.extractor)).to(device)

    sets = {}
    if args.synthetic_val:
        sets["synthetic_val"] = gather(subsample(read_jsonl(args.synthetic_val), args.synthetic_limit), "synthetic")
    if args.graded_manifest:
        sets["graded"] = gather(read_jsonl(args.graded_manifest), "graded")
    cdd = cdd11_test_records(args.test_root)
    if args.cdd11_limit_per_setting:
        seen = defaultdict(int)
        cdd = [r for r in cdd if (seen.__setitem__(r["setting"], seen[r["setting"]] + 1) or True)
               and seen[r["setting"]] <= args.cdd11_limit_per_setting]
    sets["cdd11_test"] = gather(cdd, "cdd11")

    report = {"analysis": {"config": args.config, "generator": args.generator, "extractor": args.extractor},
              "frozen": {}, "drift": {}}
    frozen_pooled = {}
    for name, items in sets.items():
        logits, pooled = run_frozen(analysis, items, device)
        frozen_pooled[name] = pooled
        report["frozen"][name] = metrics(logits, items)
        print(f"[S9] {name}: " + ", ".join(f"{k}={v:.4f}" for k, v in report["frozen"][name].items()
                                           if isinstance(v, float)), flush=True)

    if args.restorer_ckpts:
        if opt is None:
            parser.error("--restorer_ckpts needs --config (restorer architecture)")
        network_opt = dict(opt["network_g"])
        network_opt.pop("shared_extractor_path", None)
        for ckpt in args.restorer_ckpts:
            net = build_network(network_opt)
            net.load_state_dict(load_state(ckpt, args.param_key), strict=False)
            net = net.to(device).eval()
            entry = {}
            for name, items in sets.items():
                logits, similarity = run_drift(net, analysis.extractor, items, frozen_pooled[name], device)
                entry[name] = {**metrics(logits, items), "feature_similarity_to_frozen": similarity}
                print(f"[S2] {Path(ckpt).name} {name}: severity_accuracy="
                      f"{entry[name].get('severity_accuracy', float('nan')):.4f} type_f1={entry[name]['type_f1']:.4f}",
                      flush=True)
            report["drift"][str(iteration_of(ckpt))] = {"ckpt": ckpt, **entry}
            del net
            torch.cuda.empty_cache()
    write_json(args.out, report)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
