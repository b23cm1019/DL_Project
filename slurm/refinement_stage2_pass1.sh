#!/bin/bash
# Stage 2: run frozen Row C once and cache y1 for train and validation.
#SBATCH --job-name=refine_pass1
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# No fixed node: this controlled pass-1 test can run on any available GPU.
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/refine_stage2_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/refine_stage2_%j.err
#SBATCH --time=2-00:00:00
set -euo pipefail
cd /csehome/p24cs0203/krish/projects/DL_Project
source slurm/_cluster_env.sh

ROOT="${REFINEMENT_ROOT:-/scratch/p24cs0203/krish/refinement_poc}"
CHECKPOINT="${ROWC_CHECKPOINT:-/scratch/p24cs0203/krish/checkpoints/Row_C_MultiLabel_Finetune_CDD11/net_g_latest.pth}"
[[ -s "$CHECKPOINT" ]] || { echo "Missing Row C checkpoint: $CHECKPOINT" >&2; exit 2; }
[[ -s "$ROOT/manifests/train.jsonl" && -s "$ROOT/manifests/val.jsonl" ]] || { echo "Stage 1 manifests are missing" >&2; exit 2; }
PASS1_FORMAT="${PASS1_FORMAT:-jpg}"
python scripts/cache_refinement_pass1.py --manifest "$ROOT/manifests/train.jsonl" --checkpoint "$CHECKPOINT" --output-root "$ROOT/pass1" --output-manifest "$ROOT/manifests/train_pass1.jsonl" --format "$PASS1_FORMAT"
python scripts/cache_refinement_pass1.py --manifest "$ROOT/manifests/val.jsonl" --checkpoint "$CHECKPOINT" --output-root "$ROOT/pass1" --output-manifest "$ROOT/manifests/val_pass1.jsonl" --format "$PASS1_FORMAT"
wc -l "$ROOT/manifests/train_pass1.jsonl" "$ROOT/manifests/val_pass1.jsonl"
find "$ROOT/pass1" -type f | wc -l
