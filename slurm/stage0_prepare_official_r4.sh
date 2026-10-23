#!/usr/bin/env bash
# Build official CDD-11 train/val pairs and cache frozen B_improved severity.
# This is preparation only; it does not train the restorer.
#SBATCH --job-name=stage0_r4_prep
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_r4_prep_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_r4_prep_%j.err

set -euo pipefail
PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
SCRATCH_ROOT=/scratch/p24cs0203/krish
CDD_ROOT="$SCRATCH_ROOT/datasets/CDD11/train/CDD-11_train"
SPLIT_JSON="$SCRATCH_ROOT/stage0_cdd_split.json"
EXTRACTOR="$SCRATCH_ROOT/checkpoints/Stage0_B_improved/net_g_latest.pth"
ROOT="$SCRATCH_ROOT/stage0_official_r4_improved"

cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
for p in "$CDD_ROOT" "$SPLIT_JSON" "$EXTRACTOR"; do
  [[ -e "$p" ]] || { echo "Missing required path: $p" >&2; exit 2; }
done
mkdir -p "$ROOT/manifests" "$ROOT/conditions/predicted_severity"

for split in train val; do
  python scripts/build_stage0_official_cdd11_manifest.py \
    --root "$CDD_ROOT" --split-json "$SPLIT_JSON" --split "$split" \
    --output "$ROOT/manifests/${split}.jsonl"
  python scripts/cache_stage0_restoration_conditions.py \
    --manifest "$ROOT/manifests/${split}.jsonl" \
    --checkpoint "$EXTRACTOR" \
    --output-root "$ROOT/conditions/predicted_severity" \
    --output-manifest "$ROOT/manifests/${split}_predicted_severity.jsonl" \
    --mode predicted_severity --device cuda
done
echo "[RESULT] Official R4 manifests and frozen conditions prepared under $ROOT"
