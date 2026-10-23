#!/bin/bash
# Stage 3: derive frozen 16-D CLIP feedback vectors from cached y1.
#SBATCH --job-name=refine_clip
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# No fixed node: run on any available GPU with a validated project environment.
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/refine_stage3_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/refine_stage3_%j.err
#SBATCH --time=2-00:00:00
set -euo pipefail
cd /csehome/p24cs0203/krish/projects/DL_Project
source slurm/_cluster_env.sh

python - <<'PY'
import torch
import open_clip
print(f"[INFO] CLIP-ready GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] open_clip: {open_clip.__file__}")
PY

ROOT="${REFINEMENT_ROOT:-/scratch/p24cs0203/krish/refinement_poc}"
[[ -s "$ROOT/manifests/train_pass1.jsonl" && -s "$ROOT/manifests/val_pass1.jsonl" ]] || { echo "Stage 2 manifests are missing" >&2; exit 2; }
python scripts/cache_clip_feedback.py --manifest "$ROOT/manifests/train_pass1.jsonl" --output-root "$ROOT/clip" --output-manifest "$ROOT/manifests/train_clip.jsonl"
python scripts/cache_clip_feedback.py --manifest "$ROOT/manifests/val_pass1.jsonl" --output-root "$ROOT/clip" --output-manifest "$ROOT/manifests/val_clip.jsonl"
wc -l "$ROOT/manifests/train_clip.jsonl" "$ROOT/manifests/val_clip.jsonl"
find "$ROOT/clip" -name '*.npy' | wc -l
