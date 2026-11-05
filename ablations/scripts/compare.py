"""Collect evaluation results into one ablation table with paired statistics.

Reads <eval_root>/<tag>/<testset>__<probe>.json written by eval_ablation.py.

Main table (probe 'base'): CDD-11 average PSNR/SSIM, single/double/triple groups, graded in-range
and beyond-range PSNR, delta vs the reference, per-image win rate vs the reference and the
Wilcoxon signed-rank p-value over the 2,200 paired CDD-11 test images.

Probe table (--probe_tags): for each tag, every probe found, with the drop from 'base'.

  python ablations/scripts/compare.py --eval_root $OUT_ROOT/ablations/eval --ref ref_planb_200k \
      --tags dcpt_200k c2_constant s1_typeonly k1_nocal k2_nogate --probe_tags ref_planb_200k \
      --out $OUT_ROOT/ablations/tables/main.md
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(root, tag, testset, probe="base"):
    path = Path(root) / tag / f"{testset}__{probe}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def fmt(value, digits=2):
    return "–" if value is None else f"{value:.{digits}f}"


def paired(result, reference):
    if result is None or reference is None:
        return None, None
    ref = {r["id"]: r["psnr"] for r in reference["per_image"]}
    pairs = [(r["psnr"], ref[r["id"]]) for r in result["per_image"] if r["id"] in ref]
    if not pairs:
        return None, None
    a, b = np.array(pairs).T
    win = float(np.mean(a > b))
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0
    except ImportError:
        p = None
    return win, p


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--ref", required=True, help="reference tag (full model at the same budget)")
    parser.add_argument("--tags", nargs="*", default=None, help="default: every folder under eval_root")
    parser.add_argument("--probe_tags", nargs="*", default=[])
    parser.add_argument("--out", default=None, help="write the tables as markdown here")
    args = parser.parse_args()

    root = Path(args.eval_root)
    tags = args.tags if args.tags else sorted(p.name for p in root.iterdir() if p.is_dir())
    if args.ref not in tags:
        tags = [args.ref] + tags
    ref_cdd = load(root, args.ref, "cdd11")
    lines = ["| Tag | CDD-11 PSNR | SSIM | single | double | triple | graded in-range | graded beyond | Δ vs ref | win rate | Wilcoxon p |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in tags:
        cdd, graded = load(root, tag, "cdd11"), load(root, tag, "graded")
        delta = None if cdd is None or ref_cdd is None else cdd["avg_psnr"] - ref_cdd["avg_psnr"]
        win, p = (None, None) if tag == args.ref else paired(cdd, ref_cdd)
        lines.append("| " + " | ".join([
            tag + (" (ref)" if tag == args.ref else ""),
            fmt(cdd and cdd["avg_psnr"]), fmt(cdd and cdd["avg_ssim"], 4),
            fmt(cdd and cdd.get("single_psnr")), fmt(cdd and cdd.get("double_psnr")), fmt(cdd and cdd.get("triple_psnr")),
            fmt(graded and graded.get("in_range_psnr")), fmt(graded and graded.get("beyond_psnr")),
            "–" if delta is None else f"{delta:+.2f}", fmt(win, 3), "–" if p is None else f"{p:.1e}",
        ]) + " |")

    for tag in args.probe_tags:
        lines += ["", f"**Test-time probes for `{tag}`**", "",
                  "| Test set | Probe | PSNR | Δ vs base | gate | entropy |", "|---|---|---|---|---|---|"]
        for testset in ("cdd11", "graded"):
            base = load(root, tag, testset)
            if base is None:
                continue
            for path in sorted((root / tag).glob(f"{testset}__*.json")):
                result = json.loads(path.read_text())
                lines.append(f"| {testset} | {result['probe']} | {fmt(result['avg_psnr'])} | "
                             f"{result['avg_psnr'] - base['avg_psnr']:+.2f} | {fmt(result.get('gate_mean'), 3)} | "
                             f"{fmt(result.get('entropy_mean'), 3)} |")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
