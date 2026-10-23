"""Create a persistent, scene-level CDD-11 train/validation split."""
import argparse
import hashlib
import json
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="CDD-11 training root containing category folders")
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    scene_ids = set()
    for category in args.categories:
        folder = Path(args.root) / category
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        scene_ids.update(path.stem for path in folder.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    ordered = sorted(scene_ids)
    ranked = sorted(ordered, key=lambda scene: hashlib.md5(f"{args.seed}:{scene}".encode()).hexdigest())
    val_count = max(1, round(len(ranked) * args.val_fraction))
    val_ids = sorted(ranked[:val_count])
    train_ids = sorted(ranked[val_count:])
    if not train_ids or not val_ids:
        raise ValueError("Scene split must contain both train and validation scenes")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_scene_ids": train_ids,
        "val_scene_ids": val_ids,
    }, indent=2) + "\n")
    print(json.dumps({"train_scenes": len(train_ids), "val_scenes": len(val_ids), "output": str(output)}))


if __name__ == "__main__":
    main()
