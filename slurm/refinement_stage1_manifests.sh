#!/bin/bash
# Stage 1: build paired train/validation JSONL manifests.
#SBATCH --job-name=refine_manifest
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/refine_stage1_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/refine_stage1_%j.err
#SBATCH --time=04:00:00
set -euo pipefail
cd /csehome/p24cs0203/krish/projects/DL_Project
source slurm/_cluster_env.sh

ROOT="${REFINEMENT_ROOT:-/scratch/p24cs0203/krish/refinement_poc}"
TRAIN_LQ="${TRAIN_LQ:-/scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train}"
TRAIN_GT="${TRAIN_GT:-/scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train/clear}"
VAL_LQ="${VAL_LQ:-/scratch/p24cs0203/krish/datasets/CDD11/test}"
VAL_GT="${VAL_GT:-/scratch/p24cs0203/krish/datasets/CDD11/test/clear}"
for path in "$TRAIN_LQ" "$TRAIN_GT" "$VAL_LQ" "$VAL_GT"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
mkdir -p "$ROOT/manifests" "$ROOT/pass1" "$ROOT/clip" /scratch/p24cs0203/krish/logs/slurm
python scripts/build_refinement_manifest.py --lq-root "$TRAIN_LQ" --gt-root "$TRAIN_GT" --split train --output "$ROOT/manifests/train.jsonl"
python scripts/build_refinement_manifest.py --lq-root "$VAL_LQ" --gt-root "$VAL_GT" --split val --output "$ROOT/manifests/val.jsonl"
wc -l "$ROOT/manifests/train.jsonl" "$ROOT/manifests/val.jsonl"
