"""Cache full-image severity logits for an ablation that has its own Stage 2 (S5, S10, T5, X3).

Same output as scripts/cache_severity_full_conditions.py (train_full / test_full / val_subset
with inline logits), but the analysis copy and output folder are read from the Stage 3
ablation config, so cache and training can never disagree:

  analysis copy : ablation.analysis {generator, extractor, extractor_net}
  output folder : the folder of datasets.train.manifest

All 13,013 training pairs are written; held-out settings (X3) are removed by the training
dataset (exclude_settings), not here.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ablations.register  # noqa: E402,F401
from ablations.ablib.common import (  # noqa: E402
    DEFAULT_TEST_ROOT, analysis_from_config, cdd11_test_records, load_config, read_jsonl, read_rgb_tensor, write_json,
    write_jsonl,
)

CDD_MANIFESTS = Path("/scratch/p26cs0003/krish/stage0_official_r4_improved/manifests")


@torch.no_grad()
def annotate(model, rows, device, label):
    out = []
    for index, row in enumerate(rows):
        image = read_rgb_tensor(row["lq_path"])[None].to(device)
        logits = model(image)[0].float().cpu()
        out.append({"id": row["id"], "setting": row["setting"], "lq_path": row["lq_path"], "gt_path": row["gt_path"],
                    "logits": [[round(v, 5) for v in r] for r in logits.tolist()]})
        if index % 1000 == 0:
            print(f"[{label}] {index}/{len(rows)}", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="RENDERED Stage 3 ablation config")
    parser.add_argument("--cdd_manifests", default=str(CDD_MANIFESTS))
    parser.add_argument("--test_root", default=DEFAULT_TEST_ROOT)
    parser.add_argument("--val_per_setting", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0, help="debug: rows per split")
    args = parser.parse_args()

    opt = load_config(args.config)
    out = Path(opt["datasets"]["train"]["manifest"]).parent
    val_manifest = Path(opt["datasets"]["val"]["manifest"])
    if val_manifest.parent != out:
        raise ValueError("train and val manifests of the config must be in the same folder")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = analysis_from_config(opt).to(device)

    train_rows = read_jsonl(Path(args.cdd_manifests) / "train.jsonl") + read_jsonl(Path(args.cdd_manifests) / "val.jsonl")
    test_rows = cdd11_test_records(args.test_root)
    if args.limit:
        train_rows, test_rows = train_rows[:args.limit], test_rows[:args.limit]
    if len({r["id"] for r in train_rows}) != len(train_rows):
        raise ValueError("duplicate training ids")

    train = annotate(model, train_rows, device, "train")
    test = annotate(model, test_rows, device, "test")
    by_setting = defaultdict(list)
    for row in test:
        by_setting[row["setting"]].append(row)
    val_subset = [row for rows in by_setting.values() for row in rows[:args.val_per_setting]]

    write_jsonl(out / "train_full.jsonl", train)
    write_jsonl(out / "test_full.jsonl", test)
    write_jsonl(val_manifest, val_subset)
    report = {"config": args.config, "analysis": (opt.get("ablation") or {}).get("analysis"),
              "train_rows": len(train), "test_rows": len(test), "val_rows": len(val_subset)}
    write_json(out / "condition_report.json", report)
    print(report)


if __name__ == "__main__":
    main()
