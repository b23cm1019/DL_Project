"""T5: Stage 1 data rendered only from CDD-11's own clean TRAINING images (setting B).

Writes, under --out_root:
  sources/train_sources.jsonl, sources/val_sources.jsonl   clean CDD-11 train scenes (946 / 237,
      the same scene split Stage 2 uses), in the format scripts/prepare_lsdir_severity_assets.py reads
  plans/train.jsonl, plans/val.jsonl, plans/summary.json      render plans in the exact schema of
      scripts/build_lsdir_cdd11_render_plan.py (80,000 / 15,000 renders, same 11 compositions,
      severity uniform over the joint states), so the existing renderer and finalizer run unchanged.

Difference from the LSDIR plan: ~85 renders per clean image instead of at most 3, because
CDD-11 has 1,183 clean scenes instead of ~27k LSDIR images. Total render counts match setting A.
"""
import argparse
import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path

TYPES = ("low", "haze", "rain", "snow")
COMBINATIONS = (
    ("low",), ("haze",), ("rain",), ("snow",),
    ("low", "haze"), ("low", "rain"), ("low", "snow"),
    ("haze", "rain"), ("haze", "snow"),
    ("low", "haze", "rain"), ("low", "haze", "snow"),
)


def load(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def seeded_int(seed, split, relpath, index):
    return int(hashlib.sha256(f"{seed}|{split}|{relpath}|{index}".encode()).hexdigest()[:16], 16)


def sources(manifest, split):
    seen = {}
    for row in load(manifest):
        seen.setdefault(row["gt_path"], row)
    return [{"source_relpath": f"cdd11_clear/{split}/{Path(path).name}", "source_path": path} for path in sorted(seen)]


def plan(split_sources, total, split, seed):
    base, remainder = divmod(total, len(COMBINATIONS))
    compositions = [c for i, c in enumerate(COMBINATIONS) for _ in range(base + int(i < remainder))]
    rng = random.Random(f"{seed}|composition|{split}")
    rng.shuffle(compositions)
    order = list(split_sources)
    random.Random(f"{seed}|sources|{split}").shuffle(order)
    rows, per_source = [], Counter()
    for ordinal, active in enumerate(compositions):
        source = order[ordinal % len(order)]
        rows.append({"source": source, "active": active, "source_render_index": per_source[source["source_relpath"]]})
        per_source[source["source_relpath"]] += 1
    by_composition = {}
    for row in rows:
        by_composition.setdefault(row["active"], []).append(row)
    for active, group in by_composition.items():
        joint = list(itertools.product((1, 2, 3), repeat=len(active)))
        sequence = [joint[i % len(joint)] for i in range(len(group))]
        random.Random(f"{seed}|severity|{split}|{'_'.join(active)}").shuffle(sequence)
        for row, values in zip(group, sequence):
            bins = [0] * len(TYPES)
            for kind, value in zip(active, values):
                bins[TYPES.index(kind)] = value
            row["severity_bins"], row["severity_joint"] = bins, list(values)
    out = []
    for ordinal, row in enumerate(rows):
        relpath = row["source"]["source_relpath"]
        out.append({
            "schema_version": "lsdir_cdd11_render_plan_v1", "plan_ordinal": ordinal, "plan_id": f"{split}_{ordinal:06d}",
            "split": split, "source_path": row["source"]["source_path"], "source_relpath": relpath,
            "scene_id": hashlib.sha256(relpath.encode()).hexdigest()[:16],
            "source_render_index": row["source_render_index"], "composition": "_".join(row["active"]),
            "active_degradations": list(row["active"]), "severity_bins": row["severity_bins"],
            "severity_joint": row["severity_joint"], "render_seed": seeded_int(seed, split, relpath, row["source_render_index"]),
            "relative_output_path": f"images/{split}/{split}_{ordinal:06d}.png",
        })
    return out, per_source


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cdd_manifests", default="/scratch/p26cs0003/krish/stage0_official_r4_improved/manifests")
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--train_renders", type=int, default=80000)
    parser.add_argument("--val_renders", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    if (args.out_root / "plans" / "train.jsonl").exists():
        raise FileExistsError("plans already exist; keep the frozen plan (delete the folder to rebuild)")
    train_sources = sources(Path(args.cdd_manifests) / "train.jsonl", "train")
    val_sources = sources(Path(args.cdd_manifests) / "val.jsonl", "val")
    overlap = {s["source_path"] for s in train_sources} & {s["source_path"] for s in val_sources}
    if overlap:
        raise ValueError(f"{len(overlap)} clean scenes are in both the train and val CDD-11 manifests")
    write(args.out_root / "sources" / "train_sources.jsonl", train_sources)
    write(args.out_root / "sources" / "val_sources.jsonl", val_sources)
    summary = {"seed": args.seed, "setting": "B (in-protocol): CDD-11 clean training scenes only", "splits": {}}
    for split, split_sources, total in (("train", train_sources, args.train_renders), ("val", val_sources, args.val_renders)):
        rows, per_source = plan(split_sources, total, split, args.seed)
        write(args.out_root / "plans" / f"{split}.jsonl", rows)
        summary["splits"][split] = {"renders": len(rows), "clean_scenes": len(split_sources),
                                    "renders_per_scene_min": min(per_source.values()),
                                    "renders_per_scene_max": max(per_source.values()),
                                    "compositions": dict(sorted(Counter(r["composition"] for r in rows).items()))}
    (args.out_root / "plans" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
