#!/bin/bash
# Prepare manifests, frozen Row-C pass-1 outputs, and CLIP feedback caches.
# Submit this once; training jobs can depend on its successful completion.
# Override dataset paths with --export=ALL,TRAIN_LQ=...,TRAIN_GT=...,VAL_LQ=...,VAL_GT=...
# when the cluster layout differs from the defaults.
#SBATCH --job-name=refine_prepare
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/refine_prepare_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/refine_prepare_%j.err
#SBATCH --time=2-00:00:00
set -euo pipefail

cd /csehome/p24cs0203/krish/projects/DL_Project
source slurm/_cluster_env.sh

TRAIN_LQ="${TRAIN_LQ:-/scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train}"
TRAIN_GT="${TRAIN_GT:-/scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train/clear}"
VAL_LQ="${VAL_LQ:-/scratch/p24cs0203/krish/datasets/CDD11/test}"
VAL_GT="${VAL_GT:-/scratch/p24cs0203/krish/datasets/CDD11/test/clear}"
ROOT="${REFINEMENT_ROOT:-/scratch/p24cs0203/krish/refinement_poc}"
CHECKPOINT="${ROWC_CHECKPOINT:-/scratch/p24cs0203/krish/checkpoints/Row_C_MultiLabel_Finetune_CDD11/net_g_latest.pth}"

for path in "$TRAIN_LQ" "$TRAIN_GT" "$VAL_LQ" "$VAL_GT" "$CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done

mkdir -p "$ROOT/manifests" "$ROOT/pass1" "$ROOT/clip"

python scripts/build_refinement_manifest.py \
  --lq-root "$TRAIN_LQ" --gt-root "$TRAIN_GT" --split train \
  --output "$ROOT/manifests/train.jsonl"
python scripts/build_refinement_manifest.py \
  --lq-root "$VAL_LQ" --gt-root "$VAL_GT" --split val \
  --output "$ROOT/manifests/val.jsonl"

python scripts/cache_refinement_pass1.py \
  --manifest "$ROOT/manifests/train.jsonl" --checkpoint "$CHECKPOINT" \
  --output-root "$ROOT/pass1" --output-manifest "$ROOT/manifests/train_pass1.jsonl"
python scripts/cache_refinement_pass1.py \
  --manifest "$ROOT/manifests/val.jsonl" --checkpoint "$CHECKPOINT" \
  --output-root "$ROOT/pass1" --output-manifest "$ROOT/manifests/val_pass1.jsonl"

python scripts/cache_clip_feedback.py \
  --manifest "$ROOT/manifests/train_pass1.jsonl" --output-root "$ROOT/clip" \
  --output-manifest "$ROOT/manifests/train_clip.jsonl"
python scripts/cache_clip_feedback.py \
  --manifest "$ROOT/manifests/val_pass1.jsonl" --output-root "$ROOT/clip" \
  --output-manifest "$ROOT/manifests/val_clip.jsonl"

echo "Preparation complete. Counts:"
wc -l "$ROOT/manifests/train_clip.jsonl" "$ROOT/manifests/val_clip.jsonl"
find "$ROOT/pass1" -type f | wc -l
find "$ROOT/clip" -name '*.npy' | wc -l
