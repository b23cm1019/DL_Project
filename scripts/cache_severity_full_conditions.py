"""Cache full-image severity logits from the frozen Stage-2 analysis copy.

Writes (inline logits, no .npy files):
  <out>/manifests/train_full.jsonl   all 13,013 CDD-11 training pairs (train + val manifests)
  <out>/manifests/test_full.jsonl    2,200 official CDD-11 test pairs (11 settings x 200)
  <out>/manifests/val_subset.jsonl   20 test images per setting for in-training PSNR/SSIM
  <out>/condition_report.json         crop-vs-full condition gap (the Stage-3a mismatch)
                                      and optional per-primitive temperatures.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basicsr.utils.severity_full_analysis import (  # noqa: E402
    STAGE2_DIR, AnalysisCopy, cdd11_test_records, read_rgb_tensor,
)

MANIFESTS = Path("/scratch/p26cs0003/krish/stage0_official_r4_improved/manifests")
SYNTHETIC_VAL = Path("/scratch/p26cs0003/krish/severity_cdd11_lsdir_outdoor_v1/data/manifests/val.jsonl")


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@torch.no_grad()
def annotate(model, rows, device, label):
    out = []
    for index, row in enumerate(rows):
        image = read_rgb_tensor(row["lq_path"])[None].to(device)
        logits = model(image)[0].float().cpu()
        out.append({
            "id": row["id"], "setting": row["setting"], "lq_path": row["lq_path"],
            "gt_path": row["gt_path"], "logits": [[round(v, 5) for v in r] for r in logits.tolist()],
        })
        if index % 1000 == 0:
            print(f"[{label}] {index}/{len(rows)}", flush=True)
    return out


@torch.no_grad()
def crop_gap(model, rows, device, samples, crop=128, seed=0):
    """Mean |p(crop) - p(full)| and type-decision disagreement (Stage-3a mismatch)."""
    rng = random.Random(seed)
    gap = disagree = 0.0
    picked = rng.sample(rows, min(samples, len(rows)))
    for row in picked:
        full = torch.tensor(row["logits"]).softmax(-1)
        image = read_rgb_tensor(row["lq_path"])
        _, h, w = image.shape
        top, left = rng.randint(0, h - crop), rng.randint(0, w - crop)
        patch = image[:, top:top + crop, left:left + crop][None].to(device)
        cropped = model(patch)[0].float().cpu().softmax(-1)
        gap += (cropped - full).abs().sum(-1).mean().item()
        disagree += ((cropped[:, 0] < 0.5) != (full[:, 0] < 0.5)).float().mean().item()
    return {"crop_vs_full_L1_per_primitive": gap / len(picked),
            "crop_vs_full_type_disagreement": disagree / len(picked), "samples": len(picked)}


@torch.no_grad()
def fit_temperatures(model, device, limit):
    """Per-primitive temperature by NLL on synthetic val (exact severity labels)."""
    rows = read_jsonl(SYNTHETIC_VAL)[:limit]
    logits, labels = [], []
    for row in rows:
        image = read_rgb_tensor(row.get("image_path", row.get("lq_path")))[None].to(device)
        logits.append(model(image)[0].float().cpu())
        bins = row["severity_bins"]
        if isinstance(bins, dict):
            bins = [bins[name] for name in ("low", "haze", "rain", "snow")]
        labels.append(torch.tensor(bins))
    logits, labels = torch.stack(logits), torch.stack(labels)
    temperatures = []
    for t in range(logits.shape[1]):
        log_t = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits[:, t] / log_t.exp(), labels[:, t])
            loss.backward()
            return loss

        optimizer.step(closure)
        temperatures.append(round(log_t.exp().item(), 4))
    return temperatures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/scratch/p26cs0003/krish/severity_full")
    parser.add_argument("--generator", default=str(STAGE2_DIR / "net_g_latest.pth"))
    parser.add_argument("--extractor", default=str(STAGE2_DIR / "net_extractor_latest.pth"))
    parser.add_argument("--val_per_setting", type=int, default=20)
    parser.add_argument("--gap_samples", type=int, default=500)
    parser.add_argument("--temperature_fit", action="store_true")
    parser.add_argument("--temperature_samples", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0, help="debug: rows per split")
    parser.add_argument("--backbone", default="nafnet", choices=["nafnet", "restormer"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnalysisCopy(args.generator, args.extractor, backbone=args.backbone).to(device)
    out = Path(args.out)

    train_rows = read_jsonl(MANIFESTS / "train.jsonl") + read_jsonl(MANIFESTS / "val.jsonl")
    test_rows = cdd11_test_records()
    if args.limit:
        train_rows, test_rows = train_rows[:args.limit], test_rows[:args.limit]
    assert len({r["id"] for r in train_rows}) == len(train_rows), "duplicate training ids"

    train = annotate(model, train_rows, device, "train")
    test = annotate(model, test_rows, device, "test")
    by_setting = {}
    for row in test:
        by_setting.setdefault(row["setting"], []).append(row)
    val_subset = [row for rows in by_setting.values() for row in rows[:args.val_per_setting]]

    write_jsonl(out / "manifests" / "train_full.jsonl", train)
    write_jsonl(out / "manifests" / "test_full.jsonl", test)
    write_jsonl(out / "manifests" / "val_subset.jsonl", val_subset)

    report = {"backbone": args.backbone, "train_rows": len(train), "test_rows": len(test), "val_rows": len(val_subset),
              "generator": args.generator, "extractor": args.extractor}
    report.update(crop_gap(model, train, device, args.gap_samples))
    if args.temperature_fit:
        report["temperatures"] = fit_temperatures(model, device, args.temperature_samples)
    (out / "condition_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
