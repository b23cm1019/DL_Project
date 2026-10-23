#!/bin/bash
# Prepare manifests and frozen Row-C direct/CLIP condition caches for E1-E5.
# Run this once before any pilot training job.
# Override paths with --export=ALL,TRAIN_LQ=...,TRAIN_GT=...,VAL_LQ=...,VAL_GT=...
# and checkpoint paths with ROWC_G=...,ROWC_DC=....
#SBATCH --job-name=pilot_prepare
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/pilot_prepare_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/pilot_prepare_%j.err
#SBATCH --time=2-00:00:00
set -euo pipefail

PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh

ROOT="${CONDITIONED_PILOT_ROOT:-/scratch/p24cs0203/krish/conditioned_pilot}"
TRAIN_LQ="${TRAIN_LQ:-/scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train}"
TRAIN_GT="${TRAIN_GT:-/scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train/clear}"
VAL_LQ="${VAL_LQ:-/scratch/p24cs0203/krish/datasets/CDD11/test}"
VAL_GT="${VAL_GT:-/scratch/p24cs0203/krish/datasets/CDD11/test/clear}"
ROWC_G="${ROWC_G:-/scratch/p24cs0203/krish/checkpoints/Row_C_MultiLabel_Finetune_CDD11/net_g_latest.pth}"
# The original Row_C_MultiLabel_Pretrain_CDD11 directory is not present on
# this cluster. MixedCalibration is the available 4-way classifier trained
# with all CDD-11 degradation combinations and is architecture-compatible.
# Override with ROWC_DC=... if a different accepted classifier is desired.
ROWC_DC="${ROWC_DC:-/scratch/p24cs0203/krish/checkpoints/Row_C_MultiLabel_MixedCalibration_CDD11/net_dc_latest.pth}"

for path in "$TRAIN_LQ" "$TRAIN_GT" "$VAL_LQ" "$VAL_GT" "$ROWC_G" "$ROWC_DC"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done

python - <<'PY'
import open_clip
print(f"open_clip ready: {open_clip.__file__}")
PY

mkdir -p "$ROOT/manifests" "$ROOT/conditions/direct" "$ROOT/conditions/clip"

python scripts/build_refinement_manifest.py \
  --lq-root "$TRAIN_LQ" --gt-root "$TRAIN_GT" --split train \
  --output "$ROOT/manifests/train.jsonl"
python scripts/build_refinement_manifest.py \
  --lq-root "$VAL_LQ" --gt-root "$VAL_GT" --split val \
  --output "$ROOT/manifests/val.jsonl"

for split in train val; do
  python scripts/cache_rowc_pilot_conditions.py \
    --manifest "$ROOT/manifests/${split}.jsonl" \
    --rowc-checkpoint "$ROWC_G" \
    --rowc-classifier-checkpoint "$ROWC_DC" \
    --output-root "$ROOT/conditions/direct" \
    --output-manifest "$ROOT/manifests/${split}_direct.jsonl" \
    --mode direct
  python scripts/cache_rowc_pilot_conditions.py \
    --manifest "$ROOT/manifests/${split}.jsonl" \
    --rowc-checkpoint "$ROWC_G" \
    --rowc-classifier-checkpoint "$ROWC_DC" \
    --output-root "$ROOT/conditions/clip" \
    --output-manifest "$ROOT/manifests/${split}_clip.jsonl" \
    --mode clip
done

echo "Pilot preparation complete."
wc -l "$ROOT/manifests"/*_direct.jsonl "$ROOT/manifests"/*_clip.jsonl
find "$ROOT/conditions" -name '*.npy' | wc -l
