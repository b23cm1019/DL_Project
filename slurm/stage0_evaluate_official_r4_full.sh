#!/usr/bin/env bash
# Evaluate the official full-finetuned R4 checkpoint on original CDD-11 test.
# Reports overall and per-setting PSNR/SSIM for fair comparison with paper tables.
#
# Example:
#   sbatch --export=NONE,STAGE0_R4_EVAL_TAG=100k \
#     slurm/stage0_evaluate_official_r4_full.sh

#SBATCH --job-name=stage0_r4_eval
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_r4_eval_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_r4_eval_%j.err

set -euo pipefail
PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
SCRATCH_ROOT=/scratch/p24cs0203/krish
TEST_ROOT="$SCRATCH_ROOT/datasets/CDD11/test"
EXTRACTOR="$SCRATCH_ROOT/checkpoints/Stage0_B_improved/net_g_latest.pth"
TAG=${STAGE0_R4_EVAL_TAG:-100k}
RESTORER=${STAGE0_R4_EVAL_CHECKPOINT:-$SCRATCH_ROOT/checkpoints/Stage0_R4_official_full_improved/net_g_100000.pth}
ROOT="$SCRATCH_ROOT/stage0_official_cdd11_r4_full_$TAG"
MANIFEST="$ROOT/manifests/test.jsonl"
CONDITIONS="$ROOT/conditions/predicted_severity"
CONDITIONED_MANIFEST="$ROOT/manifests/test_predicted_severity.jsonl"
OUTPUT="$SCRATCH_ROOT/restoration_evaluations/Stage0_R4_official_full_improved_$TAG/test"

cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
for path in "$EXTRACTOR" "$RESTORER" "$TEST_ROOT/clear"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 3; }
done

mkdir -p "$ROOT/manifests" "$CONDITIONS"
python scripts/build_stage0_official_test_manifest.py --root "$TEST_ROOT" --output "$MANIFEST"
python scripts/cache_stage0_restoration_conditions.py \
  --manifest "$MANIFEST" --checkpoint "$EXTRACTOR" \
  --output-root "$CONDITIONS" --output-manifest "$CONDITIONED_MANIFEST" \
  --mode predicted_severity --device cuda

python scripts/evaluate_stage0_restoration.py \
  --mode R4 --checkpoint "$RESTORER" --manifest "$CONDITIONED_MANIFEST" \
  --condition-dim 16 --output "$OUTPUT" --device cuda

echo "[RESULT] Wrote $OUTPUT/summary.json"
