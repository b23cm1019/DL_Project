"""Controllability (X2): set one primitive's severity by hand and measure the restorer's response.

Uses single-primitive images of the severity-graded set (true level known). For every image the
condition of its primitive is overwritten with each assigned level 0 (absent), 1, 2, 3; the
other primitives keep the analysis copy's prediction. Reports

  psnr_matrix[true level][assigned level]   mean PSNR vs ground truth. A model that uses the
                                            condition peaks on (or next to) the diagonal.
  change_from_absent[true][assigned]        mean |output(assigned) - output(absent)|, i.e. how
                                            strongly the output reacts to the requested level.

It also saves a few example strips: input | assigned 0 | 1 | 2 | 3 | ground truth.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import ablations.register  # noqa: E402,F401
from ablations.ablib import conditions as cond  # noqa: E402
from ablations.ablib.common import (  # noqa: E402
    TYPES, analysis_from_config, build_network, image_metrics, load_config, load_state, read_jsonl, read_rgb_tensor,
    write_json,
)


def to_uint8(tensor):
    array = tensor[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return cv2.cvtColor((array * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="RENDERED Stage 3 config")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--param_key", default="params_ema")
    parser.add_argument("--graded_manifest", required=True)
    parser.add_argument("--per_cell", type=int, default=20, help="images per (primitive, true level)")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--examples", type=int, default=2, help="example strips saved per primitive")
    parser.add_argument("--out", required=True, help="output folder")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = load_config(args.config)
    network_opt = dict(opt["network_g"])
    network_opt.pop("shared_extractor_path", None)
    net = build_network(network_opt)
    net.load_state_dict(load_state(args.ckpt, args.param_key), strict=False)
    net = net.to(device).eval()
    if getattr(net, "analysis", "frozen") == "shared":
        parser.error("the shared-analysis model computes its own condition; controllability needs the frozen design")
    analysis = analysis_from_config(opt).to(device)
    transform = (opt.get("ablation") or {}).get("test_condition_transform")

    rows = [r for r in read_jsonl(args.graded_manifest) if r["composition"] in TYPES and r["level"] in args.levels]
    cells = defaultdict(list)
    for row in rows:
        if len(cells[(row["composition"], row["level"])]) < args.per_cell:
            cells[(row["composition"], row["level"])].append(row)

    psnr = defaultdict(list)
    change = defaultdict(list)
    out = Path(args.out)
    (out / "examples").mkdir(parents=True, exist_ok=True)
    saved = defaultdict(int)
    with torch.inference_mode():
        for (primitive, true_level), items in sorted(cells.items()):
            type_index = TYPES.index(primitive)
            for row in items:
                image = read_rgb_tensor(row["lq_path"])[None].to(device)
                gt_tensor = read_rgb_tensor(row["gt_path"])[None]
                gt = gt_tensor.numpy()
                logits = analysis(image)
                if transform:
                    logits = cond.apply_transform(transform, logits, row["composition"])
                outputs = {}
                for assigned in (0, 1, 2, 3):
                    output = net(image, cond.set_type_level(logits, type_index, assigned))
                    outputs[assigned] = output
                    value, _ = image_metrics(output, gt, with_ssim=False)
                    psnr[(primitive, true_level, assigned)].append(value)
                for assigned in (1, 2, 3):
                    change[(primitive, true_level, assigned)].append(
                        (outputs[assigned] - outputs[0]).abs().mean().item())
                if true_level == 2 and saved[primitive] < args.examples:
                    strip = [to_uint8(image)] + [to_uint8(outputs[a]) for a in (0, 1, 2, 3)] + [to_uint8(gt_tensor)]
                    strip = cv2.hconcat([cv2.resize(s, (s.shape[1] // 2, s.shape[0] // 2)) for s in strip])
                    cv2.imwrite(str(out / "examples" / f"{primitive}_{Path(row['gt_path']).stem}.png"), strip)
                    saved[primitive] += 1

    report = {"config": args.config, "ckpt": args.ckpt, "per_cell": args.per_cell, "primitives": {}}
    for primitive in TYPES:
        matrix, deltas = {}, {}
        for true_level in args.levels:
            if (primitive, true_level, 0) not in psnr:
                continue
            matrix[f"true_L{true_level}"] = {f"assigned_{a}": float(np.mean(psnr[(primitive, true_level, a)]))
                                             for a in (0, 1, 2, 3)}
            deltas[f"true_L{true_level}"] = {f"assigned_{a}": float(np.mean(change[(primitive, true_level, a)]))
                                             for a in (1, 2, 3)}
        report["primitives"][primitive] = {"psnr_matrix": matrix, "change_from_absent": deltas}
        print(f"== {primitive}")
        for true_key, values in matrix.items():
            print(f"  {true_key}: " + "  ".join(f"{k[-1]}:{v:6.2f}" for k, v in values.items()))
    write_json(out / "controllability.json", report)
    print(f"saved {out / 'controllability.json'}")


if __name__ == "__main__":
    main()
